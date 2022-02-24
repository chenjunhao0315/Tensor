//
//  TensorProperties.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/2/11.
//

#include "Tensor.hpp"
#include "TensorProperties.hpp"

namespace otter {

Tensor contiguous(const Tensor& self) {
    return contiguous(self, MemoryFormat::Contiguous);
}

Tensor contiguous(const Tensor& self, MemoryFormat memory_format) {
    if (self.is_contiguous(memory_format)) {
        return self;
    }
    
    return self.clone(memory_format);
}


}   // end namespace otter
