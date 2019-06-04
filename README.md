## 重构
- 重构了darknet，对于原先混杂其中的功能模块进行分离
- 重写了部分组件和功能函数，提高了运行效率以及代码可读性
- 使用基于pytorch1.1.0源码编译的libtorch库
- 使用vscode进行构建，tasks、launch等配置文件见其中

## bugs修复
- 修复由适配libtorch1.0带来的问题
- 修复tensor从文件中读取权重后造成的内存泄漏

## 帮助
- 增加darknet的所有层次结构信息，方便理解yolo的网络结构

------------------------------------

### bugs fixed for:
- !is_variable() ASSERT FAILED at ../c10/core/TensorImpl.h:922

in _Darknet.cpp_.  


__Noet:I use the libtorch by compiling from the latest source,while some errs occur with the original libtorch_yolov3 during compiling and runtime.So I do little fixes.__

About the err above,you can see more details here:https://discuss.pytorch.org/t/manually-set-variable-data-in-c/41553

------------------------------------

# libtorch-yolov3
A Libtorch implementation of the YOLO v3 object detection algorithm, written with pure C++. It's fast, easy to be integrated to your production, and CPU and GPU are both supported. Enjoy ~

This project is inspired by the [pytorch version](https://github.com/ayooshkathuria/pytorch-yolo-v3), I rewritten it with C++.

## Requirements
1. LibTorch v1.0.0
2. Cuda
3. OpenCV (just used in the example)


## To compile
1. cmake3
2. gcc 5.4 +



```
mkdir build && cd build
cmake3 -DCMAKE_PREFIX_PATH="your libtorch path" ..

# if there are multi versions of gcc, then tell cmake which one your want to use, e.g.:
cmake3 -DCMAKE_PREFIX_PATH="your libtorch path" -DCMAKE_C_COMPILER=/usr/local/bin/gcc -DCMAKE_CXX_COMPILER=/usr/local/bin/g++ ..
```


## Running the detector

The first thing you need to do is to get the weights file for v3:

```
cd models
wget https://pjreddie.com/media/files/yolov3.weights 
```

On Single image:
```
./yolo-app ../imgs/person.jpg
```

As I tested, it will take 25 ms on GPU ( 1080 ti ). please run inference job more than once, and calculate the average cost.