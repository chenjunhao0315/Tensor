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

Tensor from_rgb(const unsigned char* rgb, int h, int w, int stride);

}

#endif /* TensorPixel_hpp */
