//
//  TensorPixel.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/2/19.
//

#ifndef TensorPixel_hpp
#define TensorPixel_hpp

namespace otter {
class Tensor;

namespace cv {

Tensor from_rgb(const unsigned char* rgb, int h, int w, int stride);
Tensor from_rgba(const unsigned char* rgba, int h, int w, int stride);
Tensor from_rgba2rgb(const unsigned char* rgba, int h, int w, int stride);

}   // end namesapce cv
}   // end namespace otter

#endif /* TensorPixel_hpp */
