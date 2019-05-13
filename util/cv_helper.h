#ifndef _CV_HELPER_
#define _CV_HELPER_

#include <opencv2/opencv.hpp>
#include <stdlib.h>
#include <iostream>


/************************************
//depth info
+--------+----+----+----+----+------+------+------+------+
|        | C1 | C2 | C3 | C4 | C(5) | C(6) | C(7) | C(8) |
+--------+----+----+----+----+------+------+------+------+
| CV_8U  |  0 |  8 | 16 | 24 |   32 |   40 |   48 |   56 |
| CV_8S  |  1 |  9 | 17 | 25 |   33 |   41 |   49 |   57 |
| CV_16U |  2 | 10 | 18 | 26 |   34 |   42 |   50 |   58 |
| CV_16S |  3 | 11 | 19 | 27 |   35 |   43 |   51 |   59 |
| CV_32S |  4 | 12 | 20 | 28 |   36 |   44 |   52 |   60 |
| CV_32F |  5 | 13 | 21 | 29 |   37 |   45 |   53 |   61 |
| CV_64F |  6 | 14 | 22 | 30 |   38 |   46 |   54 |   62 |
+--------+----+----+----+----+------+------+------+------+

************************************/


#define PRINTMAT(_M) \
std::cerr << #_M <<" is:\n" <<_M<< std::endl;

#define NPF cv::Formatter::FMT_NUMPY
#define SRC_WIND "src_wind"
#define DST_WIND "dst_wind"

//can not output matrix that dims>2,
//and when dims>2,nros=ncols=-1.
#define PRINTMAT_F(_M,_F) \
if(_M.dims>2){std::cerr<<_M.size;}\
else{std::cerr << #_M <<" is:\n" <<format( _M ,_F)<<", "<<"["<<_M.rows<<","<<_M.cols<<"]" << std::endl;}

#define EXCEPTION_INFO(err) \
std::cerr<<"Exception occur:\n"<<__FILE__<<":"<<__LINE__<<"\n"<<err<<std::endl;

#define _EXIT_FAILURE \
exit(EXIT_FAILURE);

#define _CHECK_FAILURE(call) \
if(!call)\
{std::cerr<<"Error occur:\n"<<__FILE__<<":"<<__LINE__<<"\n";\
_EXIT_FAILURE;}

#define CV_TRY_CATCH(call) \
try\
{call;}\
catch(const cv::Exception& e)\
{EXCEPTION_INFO(e.what());_EXIT_FAILURE;}


#define PRINT_SZ(_M) \
if(_M.dims>2){std::cerr<<_M.size<<std::endl;}\
else{std::cerr << "["<<_M.rows<<","<<_M.cols<<"]" << std::endl;}

#define PRINT_SHAPE(_M) (PRINT_SZ(_M))

#define COND_BEGIN_ONCE do{
#define COND_END_ONCE }while(false);

#define COND_BEGIN_LOOP do{
#define COND_END_LOOP }while(true);

#define COND_PRED_BREAK(COND,PRED) \
if(COND){PRED;break;}

#define COND_PRED_THEN_BREAK(COND,PRED1,PRED2) \
if(COND){PRED;break;}else{PRED2;break;}

#define COND_PRED(COND,PRED) \
if(COND){PRED;}

#define COND_PRED_THEN(COND,PRED1,PRED2) \
if(COND){PRED1;}else{PRED2}

#define _MAIN_ARGS int argc,char **argv

#define CV_CVT2GRAY(src,dst) \
cv::Mat dst;cv::cvtColor(src,dst,cv::COLOR_BGR2GRAY);

#define CV_INIT_VECTOR(vec,count,var) \
CV_Assert(count>0);\
for(int i=0;i<count;++i)\
{vec.emplace_back(var);}

#define CV_MAT_INFO(M) \
std::cout<<#M<<" informations:\n"<<"size:"<<M.size<<"\ndims:"<<M.dims<<"\nchannels:"<<M.channels()<<"\ndepth:"<<cv_helper::get_depth(M.depth())<<std::endl;


#define xf cv::xfeatures2d
#define CV_DATA "OPENCV41_IMG_DIR"


class CV_TimeSpan
{
public:
	CV_TimeSpan();
	~CV_TimeSpan() {}
	void start();
	double stop();

	CV_TimeSpan(const CV_TimeSpan&) = delete;
	CV_TimeSpan& operator=(const CV_TimeSpan&) = delete;

private:
	int64 _start;
};

class CerrRdWrapper
{
	public:
		CerrRdWrapper(std::ostream &os);
		~CerrRdWrapper();

		CerrRdWrapper(const CerrRdWrapper&) = delete;
		CerrRdWrapper& operator=(const CerrRdWrapper&) = delete;

private:
	void *pbuf;
};

namespace cv_helper
{
	typedef void(*bar_func)(int, void*);

	namespace err
	{
#define ERR_CODE_FILESYSTEM 1
	}

	bool get_imgPath(const std::string &name,std::string &img,const char* env="IMG_FOLDER");
	bool get_imgPathEx(const std::string &name,cv::String &img, const char* env = "IMG_FOLDER");
	bool compare(const cv::Mat &m1,const cv::Mat &m2);
	void dftshift(cv::Mat &input,cv::Mat &output);
	void get_optimalImage(cv::Mat &input,cv::Mat &output);
	
	/*calculate the power spectrum density of a dft*/
	void calc_dftPSD(cv::Mat &input,cv::Mat &output,bool log=true,bool normed=false);

	/*apply filter in frequency domain*/
	void filterDomain(cv::Mat &input,cv::Mat &output,cv::Mat &filter, bool normed = false);

	/*histogram equalization for color img*/
	void histEqualization(cv::Mat &input, cv::Mat &output);

	std::string get_localTimeStr(const std::string &format="%Y_%m_%d_%H_%M_%S");
	std::string get_depth(int depth);
}

#ifdef _DEBUG

#define _DBG_OUTPUT(...) \
printf("debug info:"); \
printf(...)


#else

#define _DBG_OUPUT(...)

#endif




#endif