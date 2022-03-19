//
//  ColorConvert.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/3/19.
//

#ifndef ColorConvert_hpp
#define ColorConvert_hpp

namespace otter {

class Tensor;

namespace cv {

enum {
    RGB_TO_GRAY
};

Tensor convertColor(const Tensor& self, int mode);

}   // end namespace cv
}   // end namespace otter

#endif /* ColorConvert_hpp */
