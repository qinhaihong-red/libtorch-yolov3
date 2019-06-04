/*******************************************************************************
* 
* Author : walktree
* Email  : walktree@gmail.com
*
* A Libtorch implementation of the YOLO v3 object detection algorithm, written with pure C++. 
* It's fast, easy to be integrated to your production, and supports CPU and GPU computation. Enjoy ~
*
*******************************************************************************/
#include "Darknet.h"
#include <stdio.h>
#include <typeinfo>
#include "util/util.h"
#include <exception>
#include <stdexcept>
#include "util/cv_helper.h"
#include<fstream>
#include<algorithm>

//---------------------------------------------------------------------------
// Darknet
//---------------------------------------------------------------------------
Darknet::Darknet(const char *cfg_file, torch::Device *device) : _device(device)
{

	load_cfg(cfg_file);
	create_modules();
}

void Darknet::load_cfg(const char *cfg_file)
{
	std::ifstream fs(cfg_file);
	if (!fs)
	{
		std::cout << "Fail to load cfg file:" << cfg_file << std::endl;
		return;
	}

	std::string line;
	while (std::getline(fs, line))
	{
		std::string out;
		trim(line, out);
		line = std::move(out);

		if (line.empty() || line.front() == '#')
			continue;

		if (line.front() == '[')
		{
			Block block;
			std::string type = line.substr(1, line.length() - 2);
			block["type"] = type;
			_blocks.push_back(std::move(block));
		}
		else
		{
			std::vector<std::string> block_info;
			split(line, block_info, '=');
			if (block_info.size() == 2)
			{
				_blocks.back()[block_info[0]] = block_info[1];
			}
		}
	}
	fs.close();
}

int Darknet::get_int_from_cfg(Block &block, const std::string &key, int default_value)
{

	if (block.find(key) != block.end())
		return std::stoi(block.at(key));

	return default_value;
}

std::string Darknet::get_str_from_cfg(Block &block, const std::string &key, const std::string &default_value)
{
	if (block.find(key) != block.end())
		return block.at(key);

	return default_value;
}

Block &Darknet::get_net_info()
{
	if (_blocks.size() > 0)
		return _blocks[0];
	throw std::runtime_error("Darknet has not initiated or load cfg failure.");
}

/*
darknet模块结构：
1.共有75个conv层，其中kernel=1或3 ; stride=1或2； padding始终为1;
2.75个conv层，其中5个用于downsample，这5个conv的kernel=3,stride为2，最终下采样尺寸为2^5=32；其他conv的stride都是1
3.共有23个shortcut,都是：from=-3,activition=linear
4.共有3个yolo，4个route，2个upsample：
	yolo1(mask=6,7,8)->route(-4)->conv(k=1,out_channels=256)->upsample(stride=2)->route(-1,61)
	yolo2(mask=3,4,5)->route(-4)->conv(k=1,out_channels=128)->upsample(stride=2)->route(-1,36)
	ylol3(mask=0,1,2)


*/

void Darknet::create_modules()
{
	int input_channels = 3;
	int output_channels = 0;
	std::vector<int> output_channels_rec;

	std::cout << "begin create modules\n\n";
	//从1开始迭代，排除net block
	for (int i = 1, len = _blocks.size(); i < len; ++i)
	{
		Block &block = _blocks[i];
		std::string layer_type = block["type"];

		torch::nn::Sequential module;
		if (layer_type == "convolutional")
		{
			std::string activation = get_str_from_cfg(block, "activation", "");
			int batch_normalize = get_int_from_cfg(block, "batch_normalize", 0);
			output_channels = get_int_from_cfg(block, "filters", 0);
			int padding = get_int_from_cfg(block, "pad", 0);
			int kernel_size = get_int_from_cfg(block, "size", 0);
			int stride = get_int_from_cfg(block, "stride", 1);

			//在配置文件中，尽管所有的pad都是1，但是对于kernel_sz=1的情况，
			//pad=1是不合适的，需要置0.
			//就是说，有两种情况：
			//ksp(1,1,0) 或者 ksp(3,1,1)
			//在残差块里面，顺序是：ksp(1,1,0)+ksp(3,1,1)+shortcut(-1,-3)
			int pad = padding > 0 ? (kernel_size - 1) / 2 : 0;
			bool with_bias = batch_normalize > 0 ? false : true;

			torch::nn::Conv2d conv = torch::nn::Conv2d(get_conv_options(input_channels, output_channels, kernel_size, stride, pad, 1, with_bias));
			module->push_back(conv);

			if (batch_normalize > 0)
			{
				torch::nn::BatchNorm bn = torch::nn::BatchNorm(get_bn_options(output_channels));
				module->push_back(bn);
			}

			if (activation == "leaky")
			{
				module->push_back(torch::nn::Functional(torch::leaky_relu, /*slope=*/0.1));
			}
		}
		else if (layer_type == "shortcut")
		{
			// skip connection
			int from = get_int_from_cfg(block, "from", 0);
			block["from"] = std::to_string(from);

			// placeholder
			EmptyLayer layer;
			module->push_back(layer);
		}
		else if (layer_type == "yolo")
		{
			std::string mask_info = get_str_from_cfg(block, "mask", "");
			std::vector<int> masks;
			split(mask_info, masks, ',');

			std::string anchor_info = get_str_from_cfg(block, "anchors", "");
			std::vector<int> anchors;
			split(anchor_info, anchors, ',');

			std::vector<float> anchor_points;
			for (int j = 0; j < masks.size(); ++j)
			{
				anchor_points.push_back(anchors[masks[j] * 2]);
				anchor_points.push_back(anchors[masks[j] * 2 + 1]);
			}

			DetectionLayer layer(anchor_points,_device);
			module->push_back(layer);

			std::cout << "******create yolo********\n\n";
		}
		else if (layer_type == "route")
		{
			/* route可能是维度相加：
			route1(-4,0) :在第84层，output_c_83 = output_c_79 = 512
			route2(-1,61):在第87层，output_c_86 = output_c_85 + output_c_61 = 768
			route3(-4,0) :在第96层，output_c_95 = output_c_91 = 256
			route4(-1,36):在第99层，output_c_98 = output_c_97 + output_c_36 = 384
				
			*/
			std::string layers_info = get_str_from_cfg(block, "layers", "");

			std::vector<std::string> layers;
			split(layers_info, layers, ',');

			int start = std::stoi(layers[0]);
			int end = 0;

			if (layers.size() > 1)
				end = std::stoi(layers[1]);

			//把start 和 end 用正序表示
			if (start < 0)
				start = start + i - 1;
			if (end < 0)
				end = end + i - 1;

			block["start"] = std::to_string(start);
			block["end"] = std::to_string(end);

			output_channels = output_channels_rec[start];
			if (end > 0)
				output_channels += output_channels_rec[end];

			// placeholder
			EmptyLayer layer;
			module->push_back(layer);
		}
		else if (layer_type == "upsample")
		{
			int stride = get_int_from_cfg(block, "stride", 1);

			UpsampleLayer uplayer(stride);
			module->push_back(uplayer);
		}
		else
		{
			std::cout << "unsupported operator:" << layer_type << std::endl;
		}

		//只有conv和route会改变output_channels
		input_channels = output_channels;
		output_channels_rec.push_back(output_channels);
		_modules.push_back(module);

		char _key[50] = {0};
		sprintf(_key, "%s%d", "layer_", i - 1);

		//std::cout << "create module:" << std::string(_key) << "\n\n";

		register_module(std::string(_key), module);
	}

	//std::cout << "end create modules\n\n";
}

// #The first 4 values are header information 
// # 1. Major version number
// # 2. Minor Version Number
// # 3. Subversion number 
// # 4. IMages seen 
void Darknet::load_weights(const char *weight_file)
{
	std::ifstream fs(weight_file, std::ios::binary);
	if(!fs){throw std::runtime_error("weight file not exists.");};

	// header info: 5 * int32_t
	int32_t header_size = sizeof(int32_t) * 5;

	int64_t index_weight = 0;

	fs.seekg(0, fs.end);
	int64_t length = fs.tellg();
	// skip header
	length = length - header_size;

	fs.seekg(header_size, fs.beg);
	void *weights_src = malloc(length);
	fs.read((char *)weights_src, length);
	fs.close();

	//把权重的内容，通过from_blob写入到Tensor中.
	//注意到传入的deleter，在weights析构时，会释放内存.
	at::Tensor weights = torch::from_blob(weights_src, {length}, [](void *src) { free(src); }).toType(torch::kFloat32);

	for (int i = 0; i < _modules.size(); i++)
	{
		Block &block = _blocks[i + 1]; //block下标从i+1开始
		std::string module_type = block["type"];

		// only conv layer need to load weight
		if (module_type != "convolutional")
			continue;

		torch::nn::Sequential module = _modules[i];

		auto conv_module = module.ptr()->ptr(0);													//通过ptr(index)的方式取得子模块，属于sequential独有的
		torch::nn::Conv2dImpl *conv_imp = dynamic_cast<torch::nn::Conv2dImpl *>(conv_module.get()); //通过dynamic_cast把基类转型到子类

		int batch_normalize = get_int_from_cfg(block, "batch_normalize", 0);

		if (batch_normalize > 0)
		{
			// second module
			auto bn_module = module.ptr()->ptr(1);

			torch::nn::BatchNormImpl *bn_imp = dynamic_cast<torch::nn::BatchNormImpl *>(bn_module.get());

			int num_bn_weights = bn_imp->weight.numel();
			int num_bn_biases = bn_imp->bias.numel();
			int num_bn_running_mean = bn_imp->running_mean.numel();
			int num_bn_running_var = bn_imp->running_var.numel();

			//上面的四个num_xx的值都一样，所以，下面统一用num_bn_biases作为指针移动

			at::Tensor bn_bias = weights.slice(0, index_weight, index_weight + num_bn_biases);
			index_weight += num_bn_biases;

			at::Tensor bn_weights = weights.slice(0, index_weight, index_weight + num_bn_biases);
			index_weight += num_bn_biases;

			at::Tensor bn_running_mean = weights.slice(0, index_weight, index_weight + num_bn_biases);
			index_weight += num_bn_biases;

			at::Tensor bn_running_var = weights.slice(0, index_weight, index_weight + num_bn_biases);
			index_weight += num_bn_biases;

			bn_bias = bn_bias.view_as(bn_imp->bias);
			bn_weights = bn_weights.view_as(bn_imp->weight);
			bn_running_mean = bn_running_mean.view_as(bn_imp->running_mean);
			bn_running_var = bn_running_var.view_as(bn_imp->running_var);

			{
				torch::NoGradGuard guard;
				bn_imp->bias.copy_(bn_bias);
				bn_imp->weight.copy_(bn_weights);
				bn_imp->running_mean.copy_(bn_running_mean);
				bn_imp->running_var.copy_(bn_running_var);
			}
		}
		else
		{
			//不使用batch_norm的时候，conv才有bias

			int num_conv_biases = conv_imp->bias.numel();

			at::Tensor conv_bias = weights.slice(0, index_weight, index_weight + num_conv_biases);
			index_weight += num_conv_biases;

			conv_bias = conv_bias.view_as(conv_imp->bias);
			{
				torch::NoGradGuard guard;
				conv_imp->bias.copy_(conv_bias);
			}
		}

		int num_cov_weights = conv_imp->weight.numel();

		at::Tensor conv_weights = weights.slice(0, index_weight, index_weight + num_cov_weights);
		index_weight += num_cov_weights;

		conv_weights = conv_weights.view_as(conv_imp->weight);

		{
			torch::NoGradGuard guard;
			//即使有nograd guard，这个set_data也不行
			//conv_imp->weight.set_data(conv_weights);
			//如果不加guard，只使用copy_，会报出：a leaf Variable that requires grad has been used in an in-place operation 的错误.
			conv_imp->weight.copy_(conv_weights);

		}
	}
}

torch::Tensor Darknet::forward(torch::Tensor x)
{
	int num_modules = _modules.size();

	std::vector<torch::Tensor> outputs(num_modules);

	torch::Tensor result;
	int write = 0;

	for (int i = 0; i < num_modules; ++i)
	{
		Block &block = _blocks[i + 1];
		std::string layer_type = block["type"];

		if (layer_type == "net")
			continue;

		if (layer_type == "convolutional" || layer_type == "upsample")
		{

			//auto pre_sz = x.sizes();
			//std::cout<<"pre_sz is:\n"<<pre_sz<<"\n\n";
			x = _modules[i]->forward(x);
			outputs[i] = x;

			//dbg-info
			// if (layer_type == "upsample")
			// {
			// 	static int u = 0;
			// 	std::cout << "upsample_" << u++ << "@layer_" << i << ":\ninput_sz:" << pre_sz << ", output_sz:" << x.sizes() << "\n\n";
			// }
			// else
			// {

			// 	std::string activation = get_str_from_cfg(block, "activation", "");
			// 	int batch_normalize = get_int_from_cfg(block, "batch_normalize", 0);
			// 	int padding = get_int_from_cfg(block, "pad", 0);
			// 	int kernel_size = get_int_from_cfg(block, "size", 0);
			// 	int stride = get_int_from_cfg(block, "stride", 1);

			// 	char str[100] = {0};
			// 	sprintf(str, "(bn=%d,kernel_sz=%d,stride=%d,activation=%s)", batch_normalize, kernel_size, stride,activation.c_str());

			// 	static int c = 0;
			// 	std::cout << "conv_" << c++ << std::string(str) << "@layer_" << i << ":\ninput_sz:" << pre_sz << ", output_sz:" << x.sizes() << "\n\n";
			// }
		}
		else if (layer_type == "route")
		{
			int start = std::stoi(block["start"]);
			int end = std::stoi(block["end"]);

			if (start < 0 || end < 0)
				throw std::out_of_range("start or end index out of range!");

			x = outputs[start];
			if (end > 0)
				x = torch::cat({x, outputs[end]}, 1); //route把之前某些层的输出，进行维度相加，作为本层的输出

			outputs[i] = x;

			//dbg-info
			// static int r = 0;
			// std::cout << "route_" << r++ << "(" << start << "," << end << ")@layer_" << i << ":\noutput_sz:" << x.sizes() << "\n\n";
		}
		else if (layer_type == "shortcut")
		{
			//所有from，全部是-3
			int from = std::stoi(block["from"]);
			//auto pre_sz = x.sizes();
			x = outputs[i - 1] + outputs[i + from]; //short_cut必然是矩阵相加
			outputs[i] = x;

			//dbg-info
			// static int s = 0;
			// std::cout << "shortcut_" << s++ << "@layer_" << i << ":\ninput_sz:" << pre_sz << ", output_sz:" << x.sizes() << "\n\n";
		}
		else if (layer_type == "yolo")
		{
			Block &net_info = _blocks[0]; //net_info中的height，已经在main中进行了修正，改为416了
			int input_dim = get_int_from_cfg(net_info, "height", 0);
			int num_classes = get_int_from_cfg(block, "classes", 0);

			//auto pre_sz = x.sizes();

			x = _modules[i]->forward(x, input_dim, num_classes);

			if (write == 0)
			{
				result = x;
				write = 1;
			}
			else
			{
				result = torch::cat({result, x}, 1); //逐一把3个yolo的输出，在维度上进行叠加
			}

			//yolo层的ouput,后面的各层都不会用到： 首先route不会用到，而且后面也没有short_cut层.
			//因此仅仅是为了占位.
			outputs[i] = x;

			//dbg-info
			// static int v = 0;
			// std::cout << "yolo_" << v++ << "@layer_" << i << ":\ninput_sz:" << pre_sz << ", output_sz:" << x.sizes() << "\n";
			// std::cout << "result_sz:" << result.sizes() << "\n\n";
		}
	}
	return result;
}


torch::Tensor Darknet::write_results(torch::Tensor prediction, int num_classes, float confidence, float nms_conf)
{
	//std::ofstream log("write_results3.txt",std::ios::trunc);

	// get result which object confidence > threshold
	auto conf_mask = (prediction.select(2, 4) > confidence);//1,10647
	if(conf_mask.sum().item<int>()==0) return torch::zeros({0});
	conf_mask=conf_mask.to(torch::kFloat32).unsqueeze(2);//1,10647,1
	prediction.mul_(conf_mask);//1,10647,85
	auto non_zero_index = torch::nonzero(prediction.select(2,4)).squeeze();//过滤 0 confidence的行
	prediction = prediction.index_select(1,non_zero_index.select(1,1).squeeze());//1,n,85


	//预测坐标变换：把【中心x,中心y，宽，高】转换为【左上，右下】模式
	// top left x = centerX - w/2
	torch::Tensor box_a = torch::ones_like(prediction);
	box_a.select(2, 0) = prediction.select(2, 0) - prediction.select(2, 2).div(2);
	box_a.select(2, 1) = prediction.select(2, 1) - prediction.select(2, 3).div(2);
	box_a.select(2, 2) = prediction.select(2, 0) + prediction.select(2, 2).div(2);
	box_a.select(2, 3) = prediction.select(2, 1) + prediction.select(2, 3).div(2);
	prediction.slice(2, 0, 4) = box_a.slice(2, 0, 4);

	int batch_size = prediction.size(0);
	int item_attr_size = 5;
	bool write = false;
	int num = 0;
	torch::Tensor output = torch::zeros({0});

	for (int i = 0; i < batch_size; i++)
	{
		auto image_prediction = prediction[i];//n,85
		// get the max classes score and index at each result
		std::tuple<torch::Tensor, torch::Tensor> max_classes = torch::max(image_prediction.slice(1, item_attr_size, item_attr_size + num_classes), 1);
		auto max_conf = std::get<0>(max_classes);//n
		auto max_conf_index = std::get<1>(max_classes);//n
		max_conf = max_conf.to(torch::kFloat32).unsqueeze(1);//n,1
		max_conf_index = max_conf_index.to(torch::kFloat32).unsqueeze(1);//n,1	
		// shape: n * 7 . (left x, left y, right x, right y, object confidence, class_score, class_id)
		image_prediction = torch::cat({image_prediction.slice(1, 0, 5), max_conf, max_conf_index}, 1).reshape({-1,7}).cpu();//n,7
		

		//to unqiue class_id:过滤重复的 class id
		auto class_id = image_prediction.select(1,6).contiguous();
		std::vector<float> vec_class_id (class_id.data<float>(),class_id.data<float>()+class_id.numel());//n,
		std::vector<float> unqiue_class_id;
		std::sort(std::begin(vec_class_id),std::end(vec_class_id));
		std::unique_copy(std::begin(vec_class_id),std::end(vec_class_id),std::back_inserter(unqiue_class_id));
		
		for (auto cls_id:unqiue_class_id)
		{
			auto cls_mask = image_prediction * (image_prediction.select(1, 6) == cls_id).to(torch::kFloat32).unsqueeze(1);
			auto class_mask_index = torch::nonzero(cls_mask.select(1, 5)).squeeze();
			auto image_pred_class = image_prediction.index_select(0, class_mask_index).view({-1, 7});
			std::tuple<torch::Tensor, torch::Tensor> sort_ret = torch::sort(image_pred_class.select(1, 4));//ascending sort
			auto conf_sort_index = std::get<1>(sort_ret);
			image_pred_class = image_pred_class.index_select(0, conf_sort_index.squeeze()).cpu();//c,7

			for (int j = 0; j < image_pred_class.size(0) - 1; ++j)
			{
				int max_conf_id = image_pred_class.size(0) - 1 - j;//_c

				if (max_conf_id <= 0)break;

				auto ious = get_bbox_iou(image_pred_class[max_conf_id].unsqueeze(0), image_pred_class.slice(0, 0, max_conf_id));//_c,
				//筛选
				auto iou_mask = (ious < nms_conf).to(torch::kFloat32).unsqueeze(1);//_c,1
				image_pred_class.slice(0, 0, max_conf_id) = image_pred_class.slice(0, 0, max_conf_id) * iou_mask;
				auto non_zero_index = torch::nonzero(image_pred_class.select(1, 4)).squeeze();//__c, non_zero_index中包括max_conf_id
				image_pred_class = image_pred_class.index_select(0, non_zero_index).view({-1, 7});//__c, 7
			}

			torch::Tensor batch_index = torch::ones({image_pred_class.size(0), 1}).fill_(i);//增加一个批的维度

			if (!write)
			{
				output = torch::cat({batch_index, image_pred_class}, 1);//batch_num,class_num,8
				write = true;
			}
			else
			{
				auto out = torch::cat({batch_index, image_pred_class}, 1);
				output = torch::cat({output, out}, 0);//batch_num,class_num,8
			}

			++num;
		}//class_loop
	}//batch_loop

	if (num == 0) return torch::zeros({0});

	return output;
}