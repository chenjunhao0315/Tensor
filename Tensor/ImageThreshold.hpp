//
//  ImageThreshold.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/3/20.
//

#ifndef ImageThreshold_hpp
#define ImageThreshold_hpp

namespace otter {

class Tensor;

namespace cv {

enum {
    THRESH_BINARY = 0,
    THRESH_TRUNC = 1
};

Tensor threshold(const Tensor& self, double threshold, double maxval, int type = 0);

}   // end namespace cv
}   // end namespace otter

#endif /* ImageThreshold_hpp */
