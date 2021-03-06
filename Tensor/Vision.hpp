//
//  Vision.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/3/12.
//

#ifndef Vision_hpp
#define Vision_hpp

#include <string>
#include <vector>

enum ImreadModes {
    IMREAD_UNCHANGED = -1,
    IMREAD_GRAYSCALE = 0,
    IMREAD_COLOR = 1
};

enum ImwriteFlags {
    IMWRITE_JPEG_QUALITY = 1
};

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

// Opencv like imread function
// Note that we read rgb not bgr
Tensor imread(const std::string& path, int flags);

// Opencv like imwrite function
bool imwrite(const std::string& path, const Tensor& img, const std::vector<int>& params);

// Save image
// Note that the input should be 3-dim and HWC
int save_image(const Tensor &img, const char *name);

int save_image_jpg(const Tensor& img, const char *name, int quality = 80);

int save_image_png(const Tensor& img, const char *name);

int save_image_bmp(const Tensor& img, const char *name);

}   // end namespace cv
}   // end namespace otter

#endif /* Vision_hpp */
