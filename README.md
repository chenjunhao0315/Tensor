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

### MacOS
Build for intel
```
mkdir build && cd build
cmake ..
make -j8
```
Build for intel and M1. Note that need to revise `Config.hpp -> OTTER_MOBILE = 1` 
```
mkdir build && cd build
cmake -DCMAKE_OSX_ARCHITECTURES="x86_64;arm64" ..
make -j8
```
If you encounter `libomp` problem. Try to install `openmp` with below steps.
```
wget https://github.com/llvm/llvm-project/releases/download/llvmorg-11.0.0/openmp-11.0.0.src.tar.xz
tar -xf openmp-11.0.0.src.tar.xz
cd openmp-11.0.0.src
sed -i'' -e '/.size __kmp_unnamed_critical_addr/d' runtime/src/z_Linux_asm.S
sed -i'' -e 's/__kmp_unnamed_critical_addr/___kmp_unnamed_critical_addr/g' runtime/src/z_Linux_asm.S
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=install -DCMAKE_OSX_ARCHITECTURES="x86_64;arm64" \
            -DLIBOMP_ENABLE_SHARED=OFF -DLIBOMP_OMPT_SUPPORT=OFF -DLIBOMP_USE_HWLOC=OFF ..
cmake --build . -j 2
cmake --build . --target install
mkdir openmp-install
cp -r install/* ./openmp-install
sudo cp ./openmp-install/include/* /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/usr/include
sudo cp ./openmp-install/lib/libomp.a /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/usr/lib
```

### Linux

```
mkdir build && cd build
cmake ..
make -j 8
```

### Windows

```
g++ -Os -fopenmp -ffp-contract=fast -mavx -mavx2 -msse3 -msse4.1 -msse4.2 -msse4a -mfma -o otter *.cpp
```

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

