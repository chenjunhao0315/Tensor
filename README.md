# Tensor

## About
This is a project to implement the tensor calcuation library and (inference) neural netowrk in c++, can see it as a revision of [PyTorch][9] and [ncnn][10], the differnece is that I remove all codes that not relate to CPU, maybe will support GPU after but not now.

The netowrk structure is same as [Neural Network][11] with some enhancement and is inspired by [ConvNetJS][1], [Darknet][2], [Caffe][4] and [ncnn][10].

It aims to enhance the performance on mobile phone platform.

The main purpose of this project is used for NTHU電機系實作專題.

Add some function for image loading and saving, which is powered by [stb_image][6].

Add some drawing for image, which is powered by [OpenCV][5].

## Feature

* C++17
* No dependencies
* Multi-thread support with OpenMp
* Symobolic operation
* Arm optimization

## Documentation
* [Online Documentation](https://github.com/chenjunhao0315/Tensor/wiki)

## Build and run

### Linux, MacOS

```
mkdir build && cd build
cmake ..
make -j 8
```

### Windows

```
g++ -Os -fopenmp -mavx2 -mfma -o otter *.cpp
```

### Run

* `$ ./otter`

## Thanks for and reference
- [ConvNetjs][1]
- [Darknet][2]
- [Caffe][4]
- [PyTorch][9]
- [ncnn][10]
- [Neural Network][11]
- [Opencv][5]
- [stb_image][6]

[1]: https://cs.stanford.edu/people/karpathy/convnetjs/
[2]: https://github.com/pjreddie/darknet
[4]: https://github.com/BVLC/caffe
[5]: https://github.com/opencv/opencv
[6]: https://github.com/nothings/stb
[9]: https://github.com/pytorch/pytorch
[10]: https://github.com/Tencent/ncnn
[11]: https://github.com/chenjunhao0315/Neural_Network
