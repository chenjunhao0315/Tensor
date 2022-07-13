//
//  AffineGridGenerator.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/7/12.
//

#ifndef AffineGridGenerator_hpp
#define AffineGridGenerator_hpp

#include "Tensor.hpp"

namespace otter {

Tensor affine_grid_generator(const Tensor& theta, IntArrayRef size, bool align_corners);

}   // end namespace otter

#endif /* AffineGridGenerator_hpp */
