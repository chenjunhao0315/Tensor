//
//  Color.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/3/16.
//

#ifndef Color_hpp
#define Color_hpp

namespace otter {
namespace cv {

class Color {
public:
    Color(double v0, double v1 = 0, double v2 = 0, double v3 = 0);
    
    double val[4];
};

}   // end namepsace cv
}   // end namespace otter

#endif /* Color_hpp */
