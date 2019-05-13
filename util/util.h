#ifndef _DARKNET_UTIL_H
#define __DARKNET_UTIL_H

#include<string>
#include<vector>
#include<torch/torch.h>


// #define DBG_INFO

// #ifdef DBG_INFO
// #define DBG_BEGIN(log,path) \
// std::ofstream log(path,std::ios::trunc); 

#define D_END std::endl<<std::endl;
#define S_END std::endl;

bool  trim(const std::string &str,std::string &out);
bool split(const std::string &str,std::vector<std::string> &out,char delim=',');
bool split(const std::string &str,std::vector<int> &out,char delim=',');

torch::Tensor get_bbox_iou(torch::Tensor box1, torch::Tensor box2);

torch::nn::Conv2dOptions get_conv_options(int64_t in_planes, int64_t out_planes, int64_t kerner_size,
									  int64_t stride, int64_t padding, int64_t groups, bool with_bias = false);

torch::nn::BatchNormOptions get_bn_options(int64_t features);


class EmptyLayer : public torch::nn::Module
{
 public:   
	EmptyLayer(){};
    ~EmptyLayer(){};
	torch::Tensor forward(torch::Tensor x);
};


class UpsampleLayer : public torch::nn::Module
{
public: 
	UpsampleLayer(int stride);
    ~UpsampleLayer(){};
	torch::Tensor forward(torch::Tensor x);
private:
    int _stride;
};


class DetectionLayer : public torch::nn::Module
{
public: 
    DetectionLayer(const std::vector<float> &anchors,torch::Device *device);
    ~DetectionLayer(){};

    torch::Tensor forward(torch::Tensor prediction, int input_dim, int num_classes);
private:
    std::vector<float> _anchors;
    torch::Device *_device;
};


#endif