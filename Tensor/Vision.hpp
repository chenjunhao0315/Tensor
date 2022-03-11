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

Tensor load_image_rgb(const char* filename);

Tensor load_image_stb(const char* filename, int channels);

void cvtColor(Tensor& input, Tensor& output);

}   // end namespace cv
}   // end namespace otter

#endif /* Vision_hpp */
