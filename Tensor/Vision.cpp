//
//  Vision.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/3/12.
//

#include "Vision.hpp"

#ifndef STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#include "3rdparty/stb_image.h"
#endif
#ifndef STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "3rdparty/stb_image_write.h"
#endif

#include "Tensor.hpp"
#include "TensorMaker.hpp"
#include "TensorPixel.hpp"

namespace otter {
namespace cv {

Tensor load_image_stb(const char* filename) {
    int w, h, c;
    unsigned char *data = stbi_load(filename, &w, &h, &c, 0);
    OTTER_CHECK(data, "Cannot load image ", filename, " STB Reason: ", stbi_failure_reason());
    
    auto img = otter::from_blob(data, {h, w, c}, otter::ScalarType::Byte).clone();
    free(data);
    
    return img;
}

Tensor load_image_pixel(const char* filename) {
    return load_image_stb(filename);
}

Tensor load_image_rgb(const char* filename) {
    int w, h, c;
    unsigned char *data = stbi_load(filename, &w, &h, &c, 0);
    OTTER_CHECK(data, "Cannot load image ", filename, " STB Reason: ", stbi_failure_reason());
    
    auto img = otter::cv::from_rgb(data, h, w, w * 3);
    free(data);
    
    return img;
}

void save_image_options(const Tensor& img_, const char *name, IMG_TYPE type, int quality) {
    OTTER_CHECK(img_.dim() <= 4, "Expect the dimension of image <= 4, but get ", img_.dim());
    
    char buff[256];
    
    if (type == IMG_TYPE::PNG)      sprintf(buff, "%s.png", name);
    else if (type == IMG_TYPE::BMP) sprintf(buff, "%s.bmp", name);
    else if (type == IMG_TYPE::JPG) sprintf(buff, "%s.jpg", name);
    else                            sprintf(buff, "%s.png", name);
    
    Tensor img = img_.to(otter::ScalarType::Byte);
    int height  = (int)img.size(0);
    int width   = (int)img.size(1);
    int channel = (int)img.size(2);
    unsigned char* data = img.data_ptr<unsigned char>();
    
    int success = 0;
    if (type == IMG_TYPE::JPG)
        success = stbi_write_jpg(buff, width, height, channel, data, quality);
    
    if (!success)
        fprintf(stderr, "Failed to write image %s\n", buff);
}

void save_image_jpg(const Tensor& img, const char *name, int quality) {
    save_image_options(img, name, IMG_TYPE::JPG, quality);
}



}   // end namespace cv
}   // end namespace otter