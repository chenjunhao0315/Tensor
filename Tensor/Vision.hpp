//
//  Vision.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/3/12.
//

#ifndef Vision_hpp
#define Vision_hpp

namespace otter {
class Tensor;

namespace cv {

enum class IMG_TYPE {
    JPG,
    BMP,
    PNG
};

// Translate HWC -> NCHW
// Recommended for neural network input!
Tensor load_image_rgb(const char* filename);

// Raw data HWC
// For tranditional image process
Tensor load_image_pixel(const char* filename);

// Save image
// Note that the input should be 3-dim and HWC
void save_image(const Tensor &img, const char *name);

void save_image_jpg(const Tensor& img, const char *name, int quality = 80);

void save_image_png(const Tensor& img, const char *name);

void save_image_bmp(const Tensor& img, const char *name);

void cvtColor(Tensor& input, Tensor& output);

}   // end namespace cv
}   // end namespace otter

#endif /* Vision_hpp */
