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
#ifdef __ARM_NEON__
#define STBI_NEON
#endif

#include "TensorMaker.hpp"
#include "TensorPixel.hpp"

namespace otter {
namespace cv {

Tensor load_image_stb(const char* filename) {
    int w, h, c;
    unsigned char *data = stbi_load(filename, &w, &h, &c, 0);
    OTTER_CHECK(data, "Cannot load image ", filename, " STB Reason: ", stbi_failure_reason());
    
    auto img = otter::from_blob(data, {1, h, w, c}, otter::ScalarType::Byte).clone();
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

Tensor check_save_img_and_try_to_fix(const Tensor& img_) {
    OTTER_CHECK(img_.dim() <= 4, "Expect the dimension of image <= 4, but get ", img_.dim());
    
    Tensor img;
    
    if (img_.dim() == 4) {
        int batch_check = (int)img_.size(0);
        OTTER_CHECK(batch_check == 1, "Expect there is only one image need to be stored but get ", batch_check);
        
        img = img_.squeeze(0);
    } else {
        img = img_;
    }
    
    int channel_check_CHW = (int)img.size(0);
    int channel_check_HWC = (int)img.size(2);
    bool maybe_CHW = channel_check_CHW <= 4;
    bool maybe_HWC = channel_check_HWC <= 4;
    
    if (maybe_HWC) {    // image size too small or it is HWC
        return img.to(otter::ScalarType::Byte).contiguous();
    } else if (maybe_CHW) {
        return img.permute({1, 2, 0}).to(otter::ScalarType::Byte).contiguous();
    }
    
    OTTER_CHECK(false, "To save image, the data should be HWC or CHW");
    return Tensor();
}

void save_image_options(const Tensor& img_, const char *name, IMG_TYPE type, int quality) {
    Tensor img = check_save_img_and_try_to_fix(img_);
    
    char buff[256];
    
    if (type == IMG_TYPE::PNG)      sprintf(buff, "%s.png", name);
    else if (type == IMG_TYPE::BMP) sprintf(buff, "%s.bmp", name);
    else if (type == IMG_TYPE::JPG) sprintf(buff, "%s.jpg", name);
    else                            sprintf(buff, "%s.png", name);
    
    int height  = (int)img.size(0);
    int width   = (int)img.size(1);
    int channel = (int)img.size(2);
    
    OTTER_CHECK(channel <= 4, "Expect that the channel of image <= 4, but get ", channel);
    
    unsigned char* data = img.data_ptr<unsigned char>();
    
    int success = 0;
    if (type == IMG_TYPE::PNG)
        success = stbi_write_png(buff, width, height, channel, data, width * channel);
    else if (type == IMG_TYPE::BMP)
        success = stbi_write_bmp(buff, width, height, channel, data);
    else if (type == IMG_TYPE::JPG)
        success = stbi_write_jpg(buff, width, height, channel, data, quality);
    
    if (!success)
        fprintf(stderr, "Failed to write image %s\n", buff);
}

void save_image(const Tensor &img, const char *name) {
    save_image_jpg(img, name);
}

void save_image_jpg(const Tensor& img, const char *name, int quality) {
    save_image_options(img, name, IMG_TYPE::JPG, quality);
}

void save_image_png(const Tensor& img, const char *name) {
    save_image_options(img, name, IMG_TYPE::PNG, 100);
}

void save_image_bmp(const Tensor& img, const char *name) {
    save_image_options(img, name, IMG_TYPE::BMP, 100);
}



}   // end namespace cv
}   // end namespace otter
