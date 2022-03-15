//
//  Drawing.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/3/13.
//

#ifndef Drawing_hpp
#define Drawing_hpp

#include "GraphicAPI.hpp"

namespace otter {
class Tensor;
class Scalar;

namespace cv {

void line(Tensor& img, Point pt1, Point pt2, const Scalar& color, int thickness, int line_type, int shift);

}   // end namespace cv
}   // end namespace otter

#endif /* Drawing_hpp */
