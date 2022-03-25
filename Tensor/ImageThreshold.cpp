//
//  ImageThreshold.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/3/20.
//

#include "ImageThreshold.hpp"

#include "Tensor.hpp"
#include "Dispatch.hpp"

#include "TensorFactory.hpp"
#include "TensorFunction.hpp"

namespace otter {
namespace cv {

Tensor threshold_binary(const Tensor& self, double threshold, double true_value, double false_value);

Tensor threshold(const Tensor& self, double threshold, double maxval, int type) {
    switch (type) {
        case THRESH_BINARY:
            return threshold_binary(self, threshold, maxval, 0);
            break;
        case THRESH_TRUNC:
            return otter::native::threshold(self, threshold, maxval);
            break;
            
        default:
            break;
    }
    OTTER_CHECK(false, "Invalid threshold mode!");
    return Tensor();
}

Tensor threshold_binary(const Tensor& self, double threshold, double true_value, double false_value) {
    auto out = otter::zeros_like(self, self.scalar_type());
    
    int num_pixels = (int)(self.size(0) * self.size(1));
    
    if (self.scalar_type() == otter::ScalarType::Byte) {
        unsigned char *src = self.data_ptr<unsigned char>();
        unsigned char *ptr = out.data_ptr<unsigned char>();
        
#if __ARM_NEON
        int remain = num_pixels;
#else
        int remain = num_pixels;
#endif  // __ARM_NEON
        
        if (true_value == 255 && false_value == 0) {
#if __ARM_NEON
            
#endif
            for ( ; remain > 0; --remain) {
                *ptr = (*src > threshold) ? 255 : 0;
                
                ++src;
                ++ptr;
            }
        } else {
#if __ARM_NEON
            
#endif
            for ( ; remain > 0; --remain) {
                *ptr = (*src > threshold) ? true_value : false_value;
                
                ++src;
                ++ptr;
            }
        }
    } else {
        
    }
    
    return out;
}

}   // end namespace cv
}   // end namespace otter
