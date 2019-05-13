#include"cv_helper.h"
#include <fstream>
#include <time.h>

CV_TimeSpan::CV_TimeSpan() :_start(0)
{
	start();
}

void CV_TimeSpan::start()
{
	_start = cv::getTickCount();
}

double CV_TimeSpan::stop()
{
	return (cv::getTickCount() - _start) / cv::getTickFrequency();
}


bool cv_helper::get_imgPath(const std::string &name, std::string &img, const char* env)
{

	std::ifstream ifs(name, std::ios::binary);
	if (ifs) { img = name; return true; };

	std::string folder(getenv(env));
	if (folder.empty())
	{
		return false;
	}

	img = folder + name;
	ifs.clear(); ifs.close();
	ifs.open(img,std::ios::binary);

	auto func_open_with_appendix = [&](const std::string &app)
	{
		std::string temp = img + app;
		ifs.close();
		ifs.open(temp, std::ios::binary);
		if (ifs) { img=temp; return true; }

		return false;
	};

	if (!ifs)
	{	
		//try to add some common appendix
		if (img.rfind('.') == std::string::npos)
		{
			if (!func_open_with_appendix(".png"))
			{
				func_open_with_appendix(".jpg");
			}
		}
	}

	if (!ifs)
	{
		std::cerr << "img " << name << " not exits.\n";
		return false;
	}

	return true;
}

bool cv_helper::get_imgPathEx(const std::string &name, cv::String &img, const char* env)
{
	std::string _img;
	bool b = get_imgPath(name, _img,env);
	img = _img;

	return b;
}

std::string cv_helper::get_localTimeStr(const std::string &format)
{
	time_t t;
	time(&t);
	tm *ptm = localtime(&t);
	char buf[50] = { 0 };
	strftime(buf, sizeof(buf), format.c_str(), ptm);
	return std::string(buf);
}

CerrRdWrapper::CerrRdWrapper(std::ostream &os)
{
	pbuf = (void*)std::cerr.rdbuf();
	std::cerr.rdbuf(os.rdbuf());
}

CerrRdWrapper::~CerrRdWrapper()
{
	std::cerr.rdbuf((decltype(std::cerr.rdbuf()))pbuf);
	pbuf = nullptr;
}

bool cv_helper::compare(const cv::Mat &m1, const cv::Mat &m2)
{
	if (m1.empty() || m2.empty()) return false;
	cv::Mat com = m1 != m2;
	bool eq = cv::countNonZero(com) == 0;
	return eq;

}

std::string cv_helper::get_depth(int depth)
{
	std::string depth_name("null");
	COND_BEGIN_ONCE
		COND_PRED_BREAK(depth == CV_8U, depth_name = "CV_8U");
		COND_PRED_BREAK(depth==CV_8S,depth_name="CV_8S");
		COND_PRED_BREAK(depth==CV_16U,depth_name="CV_16U");
		COND_PRED_BREAK(depth == CV_16S, depth_name = "CV_16S");
		COND_PRED_BREAK(depth == CV_32F,depth_name="CV_32F");
		COND_PRED_BREAK(depth==CV_64F,depth_name="CV_64F");
	COND_END_ONCE

	return depth_name;
}

void cv_helper::get_optimalImage(cv::Mat &input, cv::Mat &output)
{
	int m = cv::getOptimalDFTSize(input.rows);
	int n = cv::getOptimalDFTSize(input.cols);

	cv::copyMakeBorder(input, output, 0, m - input.rows, 0, n - input.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));

}

void cv_helper::dftshift(cv::Mat &input, cv::Mat &output)
{
	output = input.clone();
	int cx = input.cols / 2;
	int cy = input.rows / 2;
	cv::Mat temp;
	output(cv::Rect(0, 0, cx, cy)).copyTo(temp);
	output(cv::Rect(cx,cy,cx,cy)).copyTo(output(cv::Rect(0, 0, cx, cy)));
	temp.copyTo(output(cv::Rect(cx, cy, cx, cy)));
	output(cv::Rect(cx,0,cx,cy)).copyTo(temp);
	output(cv::Rect(0,cy,cx,cy)).copyTo(output(cv::Rect(cx, 0, cx, cy)));
	temp.copyTo(output(cv::Rect(0, cy, cx, cy)));
}

void cv_helper::calc_dftPSD(cv::Mat &input, cv::Mat &output, bool log , bool normed )
{
	std::vector<cv::Mat> vm;
	vm.emplace_back(cv::Mat_<float>(input.clone()));
	vm.emplace_back(cv::Mat::zeros(input.size(), CV_32F));
	
	cv::Mat complexI;
	cv::merge(vm, complexI);
	cv::dft(complexI, complexI);
	cv::split(complexI, vm);

	cv::magnitude(vm[0], vm[1], output);
	if (log)
	{
		output += cv::Scalar(1);
		cv::log(output, output);
	}

	if (normed)
	{
		cv::normalize(output, output, 0, 255, cv::NORM_MINMAX);
		output.convertTo(output, CV_8U);
	}
}

void cv_helper::filterDomain(cv::Mat &input, cv::Mat &output, cv::Mat &filter, bool normed )
{
	CV_Assert(input.size() == filter.size());

	cv::Mat complexI;
	std::vector<cv::Mat> vm;
	vm.emplace_back(cv::Mat_<float>(input.clone()));
	vm.emplace_back(cv::Mat::zeros(input.size(), CV_32F));

	cv::merge(vm, complexI);
	cv::dft(complexI, complexI);

	dftshift(filter, filter);
	std::vector<cv::Mat> vm2;
	vm2.emplace_back(cv::Mat_<float>(filter.clone()));
	vm2.emplace_back(cv::Mat::zeros(filter.size(), CV_32F));
	cv::Mat complexF;
	cv::merge(vm2, complexF);

	cv::Mat dst;
	cv::mulSpectrums(complexI, complexF, dst,0);
	cv::idft(dst, dst);
	cv::split(dst, vm);

	cv::magnitude(vm[0], vm[1], output);
	if (normed)
	{
		cv::normalize(output, output, 0, 255, cv::NORM_MINMAX);
		output.convertTo(output, CV_8U);
	}
}

void cv_helper::histEqualization(cv::Mat &input, cv::Mat &output)
{
	if (input.channels()==1)
	{
		cv::equalizeHist(input, output);
	}
	else if (input.channels()==3)
	{
		std::vector<cv::Mat> imgVec;
		CV_INIT_VECTOR(imgVec, input.channels(), cv::Mat::zeros(input.size(), CV_8U));
		cv::split(input, imgVec);

		std::vector<cv::Mat> heq;
		CV_INIT_VECTOR(heq, input.channels(), cv::Mat::zeros(input.size(), CV_8U));
		cv::equalizeHist(imgVec[0], heq[0]);
		cv::equalizeHist(imgVec[1], heq[1]);
		cv::equalizeHist(imgVec[2], heq[2]);

		cv::merge(heq, output);
	}
	else
	{
		CV_Assert(!"channels>3 not supported.");
	}
}