#include "util.h"
#include <iostream>
#include <sstream>
#include <regex>
#include <exception>

bool trim(const std::string &str, std::string &out)
{
    if(str.empty()) return true;
    try
    {
        out = std::regex_replace(str, std::regex("(^ *)(.*?)( *$)"), "$2");
        return true;
    }
    catch (std::exception e)
    {
        std::cout << e.what() << std::endl;
        return false;
    }
}

bool split(const std::string &str, std::vector<std::string> &out, char delim )
{
    if (str.empty())
        return false;

    std::string item;
    std::istringstream line(str);
    while (std::getline(line, item, delim))
    {
        std::string s;
        trim(item, s);
        out.push_back(std::move(s));
    }

    return true;
}

bool split(const std::string &str, std::vector<int> &out, char delim )
{
    std::vector<std::string> vs;
    if (!split(str, vs, delim))
        return false;
    
    try
    {
        std::for_each(std::begin(vs), std::end(vs), [&out](const std::string &str) {
        out.push_back(std::stoi(str));
    });
    }
    catch(const std::exception& e)
    {
        std::cout << "split err: "<<e.what() << '\n';
    }
    
    return true;
}

// returns the IoU of two bounding boxes
torch::Tensor get_bbox_iou(torch::Tensor box1, torch::Tensor box2)
{
    // Get the coordinates of bounding boxes
    torch::Tensor b1_x1, b1_y1, b1_x2, b1_y2; //b1 shape:(1,)
    b1_x1 = box1.select(1, 0); 
    b1_y1 = box1.select(1, 1); 
    b1_x2 = box1.select(1, 2);
    b1_y2 = box1.select(1, 3);
    torch::Tensor b2_x1, b2_y1, b2_x2, b2_y2; //b2 shape:(c,)
    b2_x1 = box2.select(1, 0);
    b2_y1 = box2.select(1, 1);
    b2_x2 = box2.select(1, 2);
    b2_y2 = box2.select(1, 3);


    //Get the coordinates of the intersection rectangle
    //广播 
    torch::Tensor inter_rect_x1 = torch::max(b1_x1, b2_x1);//c,
    torch::Tensor inter_rect_y1 = torch::max(b1_y1, b2_y1);//c,
    torch::Tensor inter_rect_x2 = torch::min(b1_x2, b2_x2);//c,
    torch::Tensor inter_rect_y2 = torch::min(b1_y2, b2_y2);//c,

    // Intersection area ： 当两个box不相交时，intersetion acrea 应该为 0 . 这是下面引入torch::zeros的原因. 不要出现负值.
    //c,
    torch::Tensor inter_area = \
    torch::max(inter_rect_x2 - inter_rect_x1 + 1, torch::zeros(inter_rect_x2.sizes())) * \
    torch::max(inter_rect_y2 - inter_rect_y1 + 1, torch::zeros(inter_rect_x2.sizes()));

    // Union Area
    torch::Tensor b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1);//c,
    torch::Tensor b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1);

    torch::Tensor iou = inter_area / (b1_area + b2_area - inter_area);//c,

    return iou;
}

torch::nn::Conv2dOptions get_conv_options(int64_t in_planes, int64_t out_planes, int64_t kerner_size,
                                      int64_t stride, int64_t padding, int64_t groups, bool with_bias)
{
    torch::nn::Conv2dOptions conv_options = torch::nn::Conv2dOptions(in_planes, out_planes, kerner_size);
    conv_options.stride_ = stride;
    conv_options.padding_ = padding;
    conv_options.groups_ = groups;
    conv_options.with_bias_ = with_bias;
    return conv_options;
}

torch::nn::BatchNormOptions get_bn_options(int64_t features)
{
    torch::nn::BatchNormOptions bn_options = torch::nn::BatchNormOptions(features);
    bn_options.affine_ = true;
    bn_options.stateful_ = true;
    return bn_options;
}


//EmptyLayer

torch::Tensor EmptyLayer::forward(torch::Tensor x)
{
    return x;
}

//UpsampleLayer

UpsampleLayer::UpsampleLayer(int stride) : _stride(stride)
{
}

torch::Tensor UpsampleLayer::forward(torch::Tensor x)
{

    torch::IntArrayRef sizes = x.sizes();

    int64_t w, h;

    if (sizes.size() == 4)
    {
        w = sizes[2] * _stride;
        h = sizes[3] * _stride;

        x = torch::upsample_nearest2d(x, {w, h});
    }
    else if (sizes.size() == 3)
    {
        w = sizes[2] * _stride;
        x = torch::upsample_nearest1d(x, {w});
    }
    return x;
}

//DetectionLayer

DetectionLayer::DetectionLayer(const std::vector<float> &anchors,torch::Device *device) : _anchors(anchors),_device(device)
{
}


/*
input_dim=416*416

yolo_0@input_sz=(1,255,13,13)->output_sz=(1,507,85)
yolo_1@input_sz=(1,255,26,26)->output_sz=(1,2028,85)
yolo_2@input_sz=(1,255,52,52)->output_sz=(1,8112,85)

final_output_sz=(1,10647,85)
*/
torch::Tensor DetectionLayer::forward(torch::Tensor input, int input_dim, int num_classes)
{
    //0.确定参数与输出
    int batch_size = input.size(0);
    int stride = floor(input_dim / input.size(2));//32,16,8
    int grid_size = floor(input_dim / stride);//13,26,52
    int bbox_attrs = 5 + num_classes;//85
    int num_anchors = _anchors.size() / 2;//3

    torch::Tensor output = input.reshape({batch_size, -1, grid_size * grid_size});
    output = torch::transpose(output,1,2);
    output = output.reshape({batch_size,-1,bbox_attrs});//1,3*grid_sz*grid_sz,85


    //1.修正值域
    //select 和slice 对应于 pytorch 中的索引操作
    //对中心坐标、confidence、class score 进行sigmoid操作.
    output.select(2, 0).sigmoid_();//sig(center_x) 
    output.select(2, 1).sigmoid_();//sig(center_y)
    output.select(2, 4).sigmoid_();//sig(confidence)
    output.slice(2, 5, 5 + num_classes).sigmoid_(); //sig(classes probabilaty)

    //2.修正中心坐标
    auto grid_len = torch::arange(grid_size);
    std::vector<torch::Tensor> args = torch::meshgrid({grid_len, grid_len});
    torch::Tensor x_offset = args[1].reshape({-1, 1});//0,1,2...,0,1,2,...
    torch::Tensor y_offset = args[0].reshape({-1, 1});//0,0,0...,1,1,1,...
    x_offset = x_offset.to(*_device);
    y_offset = y_offset.to(*_device);
    auto x_y_offset = torch::cat({x_offset, y_offset}, 1).repeat({1, num_anchors}).view({-1, 2}).unsqueeze(0);//1,3*grid_sz*grid_sz,2
    /* like this:
  0  0
  0  0
  0  0
  1  0
  1  0
  1  0
  2  0
  2  0
  2  0
    */
    output.slice(2, 0, 2).add_(x_y_offset);//center_x_y + offset_x_y


    //3.修正宽高
    torch::Tensor anchors_tensor = torch::from_blob(_anchors.data(), {num_anchors, 2});//3,2
    anchors_tensor/=stride;
    anchors_tensor = anchors_tensor.to(*_device);
    //沿维度0重复 grid_size * grid_size
    //沿维度1重复 1 ，也就是不重复
    //最后再unsqueeze一下
    anchors_tensor = anchors_tensor.repeat({grid_size * grid_size, 1}).unsqueeze(0);//1,3*grid_sz*grid_sz,2
    output.slice(2, 2, 4).exp_().mul_(anchors_tensor);//(width_scale,height_scale) * (anchor_w,anchor_h)

    
    //4.统一修正尺度
    output.slice(2, 0, 4).mul_(stride);//rescale to original dimension

    return output;
}

