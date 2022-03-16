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

## Tensor
It is a tensor library which suports symbolic operation.
### Data type
Tensor support seven data types and we use `ScalarType` to indicate which data type stored in `Tensor`. Below is the conversion table
```
ScalarType::Byte -> uint8_t
ScalarType::Char -> int8_t
ScalarTtpe::Short -> int16_t
ScalarType::Int -> int
ScalarType::Long -> int64_t
ScalarType::Float -> float
ScalarType::Double -> double
```
### Initialize the tensor
#### empty
Create an empty tensor with uninitialized data.
`otter::empty(shpae, dtype)`
`otter::empty_like(Tensor)`
```c++
auto t1 = otter::empty({1, 3, 28, 28}, otter::ScalarType::Float);
auto t2 = otter::empty_like(t1);
```

#### ones
Create an tensor with all elements has value `1`
`otter::ones(shape, dtype)`
`otter::ones_like(Tensor)`
```c++
auto t1 = otter::ones({1, 3, 28, 28}, otter::ScalarType::Float);
auto t2 = otter::ones_like(t1);
```

#### zeros
Create an tensor with all elements has value `0`
`otter::zeros(shape, dtype)`
`otter::zeros_like(Tensor)`
```c++
auto t1 = otter::zeros({1, 3, 28, 28}, otter::ScalarType::Float);
auto t2 = otter::zeros_like(t1);
```

#### full
Create an tensor with all element has value we pass in
`otter::full(shpae, value, dtype)`
```c++
auto t1 = otter::full({1, 3, 28, 28}, value, otter::ScalarType::Float);
```

#### linspace
Create a tensor with linspace like numpy
`otter::linspace(start, end, steps, dtype)`
```c++
auto t1 = otter::linspace(1, 10, 10, otter::ScalarType::Float);
cout << t1 << endl;
//  1
//  2
//  3
//  4
//  5
//  6
//  7
//  8
//  9
// 10
//[ FloatType{10} ]
```

#### range
Create a tensor with range data, the difference between linspace is that the `step` in range is the `value` increase not the `division size`
`otter::range(start, end, step, dtype)`
```c++
auto t1 = otter::range(1, 10, 2, otter::ScalarType::Float);
cout << t1 << endl;
// 1
// 3
// 5
// 7
// 9
//[ FloatType{5} ]
```

#### initializer list
Create a tensor with initializer list value
`otter::tensor(initializer_list, dtype)`
```c++
auto t1 = otter::tensor({1, 4, 9}, otter::ScalarType::Float);
cout << t1 << endl;
// 1
// 4
// 9
//[ FloatType{3} ]
```

#### from blob
Create a tensor with data from outer data. **Note**: It didn't copy the data.
`otter::form_blob(data, shape, dtype)`
```c++
float data[] = {1, 4, 9};
auto t1 = otter::from_blob(data, {3}, otter::ScalarType::Float);
cout << t1 << endl;
// 1
// 4
// 9
//[ FloatType{3} ]
```
### Accessment
#### access to tensor data
Use the tensor accessor to access the tensor data
`auto accessor = Tensor.accessor<dtype, dim>();`
```c++
auto t1 = otter::ones({1, 3, 3, 3}, otter::ScalarType::Float);
auto t1_a = t1.accessor<float, 4>();

// use t1_a[][][][][] as naive multi-dimension array
t1_a[0][0][0][0] = 2;
cout << t1 << endl;
//(1,1,.,.) = 
//  2  1  1
//  1  1  1
//  1  1  1
//
//(1,2,.,.) = 
//  1  1  1
//  1  1  1
//  1  1  1
//
//(1,3,.,.) = 
//  1  1  1
//  1  1  1
//  1  1  1
//[ FloatType{1,3,3,3} ]
```

### Shape
#### view
View a tensor as different shape. **Note**: Share the same physical memory
`Tensor.view(shape) -> Tensor`
```c++
auto t1 = otter::range(1, 8, 1, otter::ScalarType::Float);
auto t2 = t1.view({1, 2, 2, 2});
cout << t1 << endl;
cout << t2 << endl;
// 1
// 2
// 3
// 4
// 5
// 6
// 7
// 8
//[ FloatType{8} ]
//(1,1,.,.) = 
//  1  2
//  3  4
//
//(1,2,.,.) = 
//  5  6
//  7  8
//[ FloatType{1,2,2,2} ]
```

#### reshape
Reshape a tensor, very like `view`. 
`Tensor.reshape(shape) -> Tensor`
```c++
auto t1 = otter::range(1, 8, 1, otter::ScalarType::Float);
auto t2 = t1.reshape({1, 2, 2, 2});
cout << t1 << endl;
cout << t2 << endl;
// 1
// 2
// 3
// 4
// 5
// 6
// 7
// 8
//[ FloatType{8} ]
//(1,1,.,.) = 
//  1  2
//  3  4
//
//(1,2,.,.) = 
//  5  6
//  7  8
//[ FloatType{1,2,2,2} ]
```

### Index
#### operator []
Use operator `[]` to index the tensor.
```c++
auto t1 = otter::range(1, 8, 1, otter::ScalarType::Float);
auto t2 = t1.view({1, 2, 2, 2});
auto t3 = t2[0][0];
auto t4 = t2[0][1];
    
cout << t2 << endl;
cout << t3 << endl;
cout << t4 << endl;
//(1,1,.,.) = 
//  1  2
//  3  4
//
//(1,2,.,.) = 
//  5  6
//  7  8
//[ FloatType{1,2,2,2} ]
// 1  2
// 3  4
//[ FloatType{2,2} ]
// 5  6
// 7  8
//[ FloatType{2,2} ]
```

#### slice
Slice the Tensor from row direction or cloumn direction. **Note**: direction: 0 -> row direction 1-> column direction
`Tensor.slice(direction, start, end, step) -> Tensor`
```c++
auto t1 = otter::range(1, 18, 1, otter::ScalarType::Float).view({-1, 6});    // auto assign dimension 0
cout << t1 << endl;
auto t2 = t1.slice(1, 2, 6);    // slice the column direction from 2 to 5
cout << t2 << endl;
auto t3 = t1.slice(0, 1, 3);    // slice the row direction from 1 to 2
cout << t3 << endl;
//  1   2   3   4   5   6
//  7   8   9  10  11  12
// 13  14  15  16  17  18
//[ FloatType{3,6} ]
//  3   4   5   6
//  9  10  11  12
// 15  16  17  18
//[ FloatType{3,4} ]
//  7   8   9  10  11  12
// 13  14  15  16  17  18
//[ FloatType{2,6} ]
```

### Permutation
#### permute
Permute the axis in Tensor.
`Tensor.permute(order) -> Tensor`
Example HWC -> CHW
```c++
auto t1 = otter::tensor({1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3}).view({2, 2, 3});
cout << t1 << endl;
auto t2 = t1.permute({2, 0, 1});
cout << t2 << endl;
//(1,.,.) = 
//  1  2  3
//  1  2  3
//
//(2,.,.) = 
//  1  2  3
//  1  2  3
//[ IntType{2,2,3} ]
//(1,.,.) = 
//  1  1
//  1  1
//
//(2,.,.) = 
//  2  2
//  2  2
//
//(3,.,.) = 
//  3  3
//  3  3
//[ IntType{3,2,2} ]
```

#### transpose
Transpose two axis.
`Tensor.transpose(dim0, dim1) -> Tensor`
```c++
auto t1 = otter::range(1, 12, 1, otter::ScalarType::Float).view({3, 4});
auto t2 = t1.transpose(1, 0);
cout << t1 << endl;
cout << t2 << endl;
//  1   2   3   4
//  5   6   7   8
//  9  10  11  12
//[ FloatType{3,4} ]
//  1   5   9
//  2   6  10
//  3   7  11
//  4   8  12
```

### Type conversion
Convert data dtype in Tensor with casting.
`Tensor.to(dtype) -> Tensor`
```c++
auto t1 = otter::tensor({1.5, 2.5, 3.5});
cout << t1 << endl;
auto t2 = t1.to(otter::ScalarType::Int);
cout << t2 << endl;
// 1.5000
// 2.5000
// 3.5000
//[ DoubleType{3} ]
// 1
// 2
// 3
//[ IntType{3} ]
```

### Copy
#### clone
Use `.clone()` to make a deep copy.
`Tensor.clone() -> Tensor`
```c++
auto t1 = otter::empty({1}, otter::ScalarType::Float);
auto t2 = t1;    // shadow copy
auto t3 = t1.clone();    // deep copy
cout << "t1 physical address: " << t1.data_ptr<float>() << endl;    // original data
cout << "t2 physical address: " << t2.data_ptr<float>() << endl;    // shadow copy
cout << "t3 physical address: " << t3.data_ptr<float>() << endl;    // deep copy
// t1 physical address: 0x105c08140
// t2 physical address: 0x105c08140
// t3 physical address: 0x105c0a140
```

### Operation
#### add
Element-wise addition
##### member function
```
add(Tensor) -> Tensor
add(Scalar) -> Tensor
add_(Tensor) -> Tensor&    // inplace
add_(Scalar) -> Tensor&    // inplace
```
```c++
auto t1 = otter::range(1, 10, 2, otter::ScalarType::Float);
auto t2 = otter::range(11, 20, 2, otter::ScalarType::Float);
auto t3 = t1.add(t2);
cout << t3 << endl;
t3.add_(2);
cout << t3 << endl;
// 12
// 16
// 20
// 24
// 28
//[ FloatType{5} ]
// 14
// 18
// 22
// 26
// 30
//[ FloatType{5} ]
```
##### binaray operation
```
TensorC = TensorA + TensorB
```
```c++
auto t1 = otter::range(1, 10, 2, otter::ScalarType::Float);
auto t2 = otter::range(11, 20, 2, otter::ScalarType::Float);
auto t3 = t1 + t2;
cout << t3 << endl;
// 12
// 16
// 20
// 24
// 28
//[ FloatType{5} ]
```

#### sub
Element-wise substraction
##### member function
```c++
sub(Tensor) -> Tensor
sub(Scalar) -> Tensor
sub_(Tensor) -> Tensor&    // inplace
sub_(Scalar) -> Tensor&    // inplace
```
##### binaray operation
```c++
TensorC = TensorA - TensorB
```

#### mul
Element-wise multiplication
##### member function
```c++
mul(Tensor) -> Tensor
mul(Scalar) -> Tensor
mul_(Tensor) -> Tensor&    // inplace
mul_(Scalar) -> Tensor&    // inplace
```
##### binaray operation
```c++
TensorC = TensorA * TensorB
```

#### div
Element-wise divition
##### member function
```c++
div(Tensor) -> Tensor
div(Scalar) -> Tensor
div_(Tensor) -> Tensor&    // inplace
div_(Scalar) -> Tensor&    // inplace
```
##### binaray operation
```c++
TensorC = TensorA / TensorB
```

#### remainder
Element-wise remainder
##### member function
```c++
remainder(Tensor) -> Tensor
remainder(Scalar) -> Tensor
remainder_(Tensor) -> Tensor&    // inplace
remainder_(Scalar) -> Tensor&    // inplace
```
##### binaray operation
```c++
TensorC = TensorA % TensorB
```

#### bitwise and
Bit-wise and
##### member function
```c++
bitwise_and(Tensor) -> Tensor
bitwise_and(Scalar) -> Tensor
bitwise_and_(Tensor) -> Tensor&    // inplace
bitwise_and_(Scalar) -> Tensor&    // inplace
```
##### binaray operation
```c++
TensorC = TensorA & TensorB
```

#### bitwise or
Bit-wise or
##### member function
```c++
bitwise_or(Tensor) -> Tensor
bitwise_or(Scalar) -> Tensor
bitwise_or_(Tensor) -> Tensor&    // inplace
bitwise_or_(Scalar) -> Tensor&    // inplace
```
##### binaray operation
```c++
TensorC = TensorA & TensorB
```

#### bitwise xor
Bit-wise xor
##### member function
```c++
bitwise_xor(Tensor) -> Tensor
bitwise_xor(Scalar) -> Tensor
bitwise_xor_(Tensor) -> Tensor&    // inplace
bitwise_xor_(Scalar) -> Tensor&    // inplace
```
##### binaray operation
```c++
TensorC = TensorA ^ TensorB
```

#### bitwise not
Bit-wise not
##### member function
```c++
bitwise_not(Tensor) -> Tensor
bitwise_not(Scalar) -> Tensor
bitwise_not_(Tensor) -> Tensor&    // inplace
bitwise_not_(Scalar) -> Tensor&    // inplace
```
##### Unary operation
```c++
TensorB = ~TesorA
```

#### neg
Element-wise neg
##### member function
```c++
neg() -> Tensor
neg_() -> Tensor&
```
##### Unary operation
```c++
TensorB = -TesorA
```

#### abs
Element-wise abs
##### member function
```c++
abs() -> Tensor
abs_() -> Tensor&
```
##### Unary operation
```c++
TensorB = otter::abs(TesorA)
```

#### sin
Element-wise sin
##### member function
```c++
sin() -> Tensor
sin_() -> Tensor&
```
##### Unary operation
```c++
TensorB = otter::sin(TesorA)
```

#### cos
Element-wise cos
##### member function
```c++
cos() -> Tensor
cos_() -> Tensor&
```
##### Unary operation
```c++
TensorB = otter::cos(TesorA)
```

#### tan
Element-wise tan
##### member function
```c++
tan() -> Tensor
tan_() -> Tensor&
```
##### Unary operation
```c++
TensorB = otter::tan(TesorA)
```
#### exp
Element-wise exp
##### member function
```c++
exp() -> Tensor
exp_() -> Tensor&
```
##### Unary operation
```c++
TensorB = otter::exp(TesorA)
```

#### sqrt
Element-wise sqrt
##### member function
```c++
sqrt() -> Tensor
sqrt_() -> Tensor&
```
##### Unary operation
```c++
TensorB = otter::sqrt(TesorA)
```

### dot
Dot operation. **Note**: 1D tensor, same data type, same data size
##### member function
```c++
dot(Tensor) -> Tensor
```

## Image I/O
Use `stb_image` library

### Load image
Load image from `*.jpg`, `*.png`, ...

* `load_image_pixel(filename);`    Load image from file and store the data in HWC format(**d = 3**)
* `load_image_rgb(filename);` Load image from file and store the data in NCHW format(**d = 4**)

```c++
auto img_hwc = otter::cv::load_image_pixel("hello_world.jpg");    // Memory format: HWC -> for tranditional image processing

auto img_nchw = otter::cv::load_image_rgb("hello_world.jpg");    // Memory format: NCHW -> for neural network input
```

### Save Image
Save image into `*.jpg`, `*.png`, `*.bmp`

* `save_image(img, filename);` Save image in `jpg` format for convience
* `save_image_jpg(img, filename, quality);`    Save image in `jpg` format
* `save_image_png(img, filename);`    Save image in `png` format
* `save_image_bmp(img, filename);`    Save image in `bmp` format

```c++
auto img = otter::cv::load_image_pixel("hello_world.jpg");    // Load image

otter::cv::save_image(img, "test");
```

## Drawing
Now use the code from [OpenCV][5], all the function work the same as it
#### Line
Draw line on image
* `line(img,  point1, point2, color, thickness = 1, line_type = LINE_8, shift = 0);`    Draw a straight line form point1 to point2 using `Bresenham algorithm`

```c++
auto img = otter::zeros({50, 50, 3}, otter::ScalarType::Byte);    // Create an empty 50*50 canvas with three channels

otter::cv::line(img, otter::cv::Point(10, 10), otter::cv::Point(40, 40), otter::cv::Color(255, 0, 0), 3);    // Draw a line form (10, 10) to (40, 40) with rgb(255, 0, 0) and thickness is 3 using default line_type and default shift

otter::cv::save_image(img, "test");
```

#### Circle
Draw circle on image
* `circle(img, center, radius, color, thickness = 1, line_type = LINE_8, shift = 0)`    Draw a circle with given center and radius

```c++
auto img = otter::zeros({50, 50, 3}, otter::ScalarType::Byte);    // Create an empty 50*50 canvas with three channels

otter::cv::circle(img, otter::cv::Point(25, 25), 10, otter::cv::Color(255, 0, 0), 3);    // Draw a circle at (25, 25) radius(10) with rgb(255, 0, 0) and thickness is 3 using default line_type and default shift

otter::cv::save_image(img, "test");
```

#### Rectangle
Draw rectangle on image
* `rectangle(img, point1, point2, color, thickness = 1, line_type = LINE_8, shift = 0);`    Draw a rectangle with left top point1 to bottom right point2
* `rectangle(img, rect, color, thickness = 1, line_type = LINE_8, shift = 0);`    Draw a rectangle with given rectangle position

```c++
auto img = otter::zeros({50, 50, 3}, otter::ScalarType::Byte);    // Create an empty 50*50 canvas with three channels

// Both two method create identical result

// Method 1
otter::cv::rectangle(img, otter::cv::Point(10, 10), otter::cv::Point(40, 40), otter::cv::Color(255, 0, 0), 3);    // Draw a rectangle form (10, 10) to (40, 40) with rgb(255, 0, 0) and thickness is 3 using default line_type and default shift

// Method 2
otter::cv::rectangle(img, otter::cv::Rect(10, 10, 30, 30), otter::cv::Color(255, 0, 0), 3);    // Draw a rectangle form (10, 10) with width (30, 30) with rgb(255, 0, 0) and thickness is 3 using default line_type and default shift

otter::cv::save_image(img, "test");
```

#### Put Text
Put text on image
* `putText(img, string, point, font_face, font_scale, color, thickness = 1, line_type = LINE_8, shift = 0)`    Put text from left top point with given font_face and font_scale
* `putText(img, string, point, font_face, pixelHeight, color, thickness = 1, line_type = LINE_8, shift = 0)`    Put text from left top point with given font_face and pixelHeight

##### Font face

* `FONT_HERSHEY_SIMPLEX`    normal size sans-serif font
* `FONT_HERSHEY_PLAIN`        small size sans-serif font
* `FONT_HERSHEY_DUPLEX`        normal size sans-serif font (more complex than FONT_HERSHEY_SIMPLEX)
* `FONT_HERSHEY_COMPLEX`    normal size serif font
* `FONT_HERSHEY_TRIPLEX`    normal size serif font (more complex than FONT_HERSHEY_COMPLEX)
* `FONT_HERSHEY_COMPLEX_SMALL`  smaller version of FONT_HERSHEY_COMPLEX
* `FONT_HERSHEY_SCRIPT_SIMPLEX` hand-writing style font
* `FONT_HERSHEY_SCRIPT_COMPLEX` more complex variant of FONT_HERSHEY_SCRIPT_SIMPLEX
* `FONT_ITALIC` flag for italic font


```c++
auto img = otter::zeros({500, 500, 3}, otter::ScalarType::Byte);    // Create an empty 500*500 canvas with three channels
    
otter::cv::line(img, otter::cv::Point(0, 0), otter::cv::Point(250, 250), otter::cv::Color(255, 0, 0), 20, 8, 0);    // Draw line from (0, 0) to (250, 250)
    
otter::cv::circle(img, otter::cv::Point(250, 250), 50, otter::cv::Color(0, 255, 0), 5, otter::cv::LINE_8, 0);    // Draw circle at (250, 250) with radius 50
    
otter::cv::rectangle(img, otter::cv::Point(100, 100), otter::cv::Point(400, 400), otter::cv::Color(0, 0, 255), 10, 8, 0);    // Draw rectangle from (100, 100) to (400, 400)
    
otter::cv::putText(img, "test", otter::cv::Point(100, 100), otter::cv::FONT_HERSHEY_SIMPLEX, 50, otter::cv::Color(255, 0, 0), 3, otter::cv::LINE_AA, false);    // Put text at (100, 100)
    
otter::cv::save_image_jpg(img, "test", 100);
```

## Supported Layers

#### Data layer

* Input layer (raw data input)

#### Vision layers

* Convolution layer (depthwise support)
* MaxPool layer
* UpSample layer

#### Common layers

* Dropout layer

#### Activation layers

* LRelu layer

#### Normalization layer

* BatchNormalization layer

#### Utility layers

* Concat layer (multi layers)
* ShortCut layer (single layer)

#### Loss layers

#### Special layers

## Supported Trainers

## Supported Model

## Construct network

#### Initialize network 

Declear the nerual network.

```cpp
Net nn;
```

#### Add layers

It will add layer to neural network, checking the structure of input tensor at some layer, for instance, Concat layer, the input width and height should be the same as all input.

```cpp
nn.addLayer(LayerOption{{"type", "XXX"}, {"option", "YYY"}, {"input", "ZZZ"}, {"name", "WWW"}});    // The options are unordered
```

##### Data layer

* Input layer options

> **channel** <br>
> **width** <br>
> **height**

##### Vision layers

* Convolution layer options

> **out_channel** <br>
> kernel <br>
> kernel_width <br>
> kernel_height <br>
> stride (1) <br>
> stride_height (-1) <br>
> stride_width (-1) <br>
> dilation (1) <br>
> dilation_height (1) <br>
> dilation_width (1) <br>
> padding (0) <br>
> padding_height (0) <br>
> padding_width (0) <br>
> groups (1) <br>
> batchnorm (none) <br>
> activation (none)

* MaxPool layer

> kernel <br>
> kernel_width <br>
> kernel_height <br>
> stride (1) <br>
> stride_height (-1) <br>
> stride_width (-1) <br>
> dilation (1) <br>
> dilation_height (1) <br>
> dilation_width (1) <br>
> padding (0) <br>
> padding_height (0) <br>
> padding_width (0) <br>
> ceil_mode (false) (ceil or not when calculating output shape) <br>
> darknet_mode(flase) (ensure the pooling behaviour is same as darknet)

* AvgPooling layer
* UpSample layer

> **stride**

##### Common layers

* Dropout layer

> probability (0.5)

##### Activation layers

* LRelu layer

> alpha (1)

##### Normalization layer

* BatchNormalization layer

##### Utility layers

* Concat layer (multi layers)

> input (list of input name) <br>

* ShortCut layer (single layer)

> input (list of input name) <br>

##### Loss layers

##### Special layers

#### Construct network

It will construct the static computation graph of neural network automatically, but not checking is it reasonable or not.

```cpp
nn.compile();
```

#### Network summary

Show the breif detail between layer and layer.

```cpp
nn.summary();    // Show network shape
```

#### Forward Propagation

The data flow of network is based on **Tensor**. To forward propagation, we need to create a extractor first and pass the input data in. Then use the extractor to extract the result.

```cpp
Tensor input = otter::ones({1, 3, 28, 28}, otter::ScalarType::Float);

auto extractor = nn.create_extractor();    // create an extractor
extractor.input("data", input);    // pass input data in
Tensor output;    // tensor that output will be stored
extractor.extract("output", output, 0);    // extract the result
```

#### Backward Propagation

#### Save otter model &hearts;

#### Load otter model

## Construct trainer

#### Initialze the trainer

###### Trainer options

###### Learning rate policy

###### Warmup

#### Start training

## Build and run

#### Linux, MacOS

```
cmake .
cmake --build .
```

#### Windows

```
g++ -Os -fopenmp -mavx2 -o otter *.cpp
```

#### Run

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

