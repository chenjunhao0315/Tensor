//
//  TensorPacked.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/5/11.
//

#include "TensorPacked.hpp"
#include "TensorFactory.hpp"

namespace otter {

Tensor convertElempack(const Tensor& self, int64_t out_elempack, bool use_padding)
{
    OTTER_CHECK(self.dim() <= 4, "Unsupport convert packing!");
    
//    int64_t elempack = self.elempack();
//    
//    if (elempack == out_elempack) {
//        return self;
//    }
//    
//    int64_t w = 0;
//    int64_t h = 0;
//    int64_t c = 0;
//    int64_t b = 0;
//    int64_t dims = self.dim();
//    int64_t itemsize = self.itemsize();
//    
//    if (dims == 1) {
//        w = self.size(0);
//    } else if (dims == 2) {
//        w = self.size(1);
//        h = self.size(0);
//    } else if (dims == 3) {
//        w = self.size(2);
//        h = self.size(1);
//        c = self.size(0);
//    } else if (dims == 4) {
//        w = self.size(3);
//        h = self.size(2);
//        c = self.size(1);
//        b = self.size(0);
//    }
//    
//    if (!use_padding) {
//        if (dims == 1 && w * elempack % out_elempack != 0) {
//            return self;
//        }
//        if (dims == 2 && h * elempack % out_elempack != 0) {
//            return self;
//        }
//        if ((dims == 3 || dims == 4) && c * elempack % out_elempack != 0) {
//            return self;
//        }
//    }
    
    return self;
}

}   // end namespace otter
