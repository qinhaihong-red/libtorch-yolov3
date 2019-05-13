/*******************************************************************************
* 
* Author : walktree
* Email  : walktree@gmail.com
*
* A Libtorch implementation of the YOLO v3 object detection algorithm, written with pure C++. 
* It's fast, easy to be integrated to your production, and supports CPU and GPU computation. Enjoy ~
*
*******************************************************************************/

#include <torch/torch.h>
#include <string>
#include <vector>
#include <map>


using Block = std::map<std::string ,std::string>;

class Darknet : public torch::nn::Module
{

public:
	Darknet(const char *conf_file, torch::Device *device);
	~Darknet(){};

	// load YOLOv3
	void 		  load_cfg(const char *cfg_file);
	void 		  load_weights(const char *weight_file);
	Block&		  get_net_info();
	int 		  get_int_from_cfg(Block &block, const std::string &key, int default_value);
	std::string   get_str_from_cfg(Block &block, const std::string &key, const std::string &default_value);

	void 		  create_modules();
	torch::Tensor forward(torch::Tensor x);
	/* 对预测数据进行筛选*/
	torch::Tensor write_results(torch::Tensor prediction, int num_classes, float confidence, float nms_conf = 0.4);

private:
	torch::Device 					  *_device;
	std::vector<Block> 				   _blocks;
	std::vector<torch::nn::Sequential> _modules;
};