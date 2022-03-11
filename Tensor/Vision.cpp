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
#include "TensorPixel.hpp"

namespace otter {
namespace cv {

Tensor load_image_stb(const char* filename, int channels) {
    int w, h, c;
    unsigned char *data = stbi_load(filename, &w, &h, &c, channels);
    if (!data) {
        
    }
    if(channels) c = channels;
    auto img = otter::cv::from_rgb(data, h, w, w * 3);
    
    return img;
}

Tensor load_image_rgb(const char* filename) {
    return load_image_stb(filename, 3);
}



}   // end namespace cv
}   // end namespace otter
