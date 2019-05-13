#include <torch/torch.h>
#include <iostream>
#include <chrono>
#include <time.h>
#include "opencv2/opencv.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "Darknet.h"
#include "util/util.h"

using namespace std;
using namespace std::chrono;

int main(int argc, const char *argv[])
{
    // if (argc != 2)
    // {
    //     std::cerr << "usage: yolo-app <image path>\n";
    //     return -1;
    // }

    torch::DeviceType device_type;

    if (torch::cuda::is_available())
    {
        device_type = torch::kCUDA;
    }
    else
    {
        device_type = torch::kCPU;
    }
    torch::Device device(device_type);

    // input image size for YOLO v3
    int input_image_size = 416;

    Darknet net("/home/hh/deeplearning_daily/libtorch-yolov3/models/yolov3.cfg", &device);

    Block &net_info = net.get_net_info();
    net_info["height"] = std::to_string(input_image_size);//配置文件里面height可能不是416,这里要再修正一下

    std::cout << "loading weight ..." << endl;
    net.load_weights("/home/hh/deeplearning_daily/darknet_src/darknet/yolov3.weights");
    std::cout << "weight loaded ..." << endl;

    net.to(device);

    torch::NoGradGuard no_grad;
    net.eval();

    std::cout << "start to inference ..." << endl;

    cv::Mat origin_image, resized_image;

    origin_image = cv::imread("/home/hh/deeplearning_daily/libtorch-yolov3/imgs/dog.jpg");
    //origin_image = cv::imread(argv[1]);

    cv::resize(origin_image, resized_image, cv::Size(input_image_size, input_image_size));

    cv::Mat img_float;
    resized_image.convertTo(img_float, CV_32F, 1.0 / 255);

    auto img_tensor = torch::from_blob(img_float.data, {1, input_image_size, input_image_size, 3});
    img_tensor = img_tensor.toType(torch::kFloat32);
    img_tensor = img_tensor.permute({0, 3, 1, 2});
    auto img_var = img_tensor.to(device);

    auto start = std::chrono::high_resolution_clock::now();

    auto output = net.forward(img_var);
    std::cout<<"output size is:"<<output.sizes()<<std::endl;//1,10647,85

    // filter result by NMS
    // class_num = 80
    // confidence = 0.6
    auto result = net.write_results(output, 80, 0.6, 0.4);

    auto end = std::chrono::high_resolution_clock::now();

    auto duration = duration_cast<milliseconds>(end - start);

    // It should be known that it takes longer time at first time
    std::cout << "inference taken : " << duration.count() << " ms" << endl;

    if (result.dim() == 1)
    {
        std::cout << "no object found" << endl;
    }
    else
    {
        std::cout<<"result size is:\n"<<result.sizes()<<D_END;
        std::cout<<"result is:\n"<<result<<D_END;



        int obj_num = result.size(0);

        std::cout << obj_num << " objects found" << D_END;

        float w_scale = float(origin_image.cols) / input_image_size;
        float h_scale = float(origin_image.rows) / input_image_size;


        std::cout<<"w_scale and h_scale is:\n"<<w_scale<<" ,"<<h_scale<<D_END;

        //二次尺度修正
        result.select(1, 1).mul_(w_scale);//从416恢复到原始尺度
        result.select(1, 2).mul_(h_scale);
        result.select(1, 3).mul_(w_scale);
        result.select(1, 4).mul_(h_scale);

        std::cout<<"result after scale:\n"<<result<<D_END;
        auto result_data = result.accessor<float, 2>();

        std::cout<<"result_data size is:\n"<<result_data.sizes()<<D_END;

        for (int i = 0; i < result.size(0); i++)
        {
            cv::rectangle(origin_image, cv::Point(result_data[i][1], result_data[i][2]), cv::Point(result_data[i][3], result_data[i][4]), cv::Scalar(0, 0, 255), 1, 1, 0);

            // std::cout<<result_data[i][1]<<" ,"<<result_data[i][2]<<" ,"<<result_data[i][3]<<" ,"<<result_data[i][4]<<std::endl;
            // std::cout<<result[i][1]<<" ,"<<result[i][2]<<" ,"<<result[i][3]<<result[i][4]<<std::endl<<std::endl;
        }

        cv::imwrite("out-det.jpg", origin_image);
    }

    std::cout << "\nDone" << endl;

    return 0;
}