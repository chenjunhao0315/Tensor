//
//  LineIterator.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/3/14.
//

#ifndef LineIterator_hpp
#define LineIterator_hpp

#include "GraphicAPI.hpp"

namespace otter {
class Tensor;

namespace cv {

class LineIterator {
public:
    LineIterator(const Tensor& img, Point pt1, Point pt2);
private:
};

}   // end namespace cv
}   // end namespace otter

#endif /* LineIterator_hpp */
