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
    PNG,
    
};

Tensor load_image_pixel(const char* filename);

Tensor load_image_rgb(const char* filename);

void save_image_jpg(const Tensor& im, const char *name, int quality = 80);

void cvtColor(Tensor& input, Tensor& output);

}   // end namespace cv
}   // end namespace otter

#endif /* Vision_hpp */
