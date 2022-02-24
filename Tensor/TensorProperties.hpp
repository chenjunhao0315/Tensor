//
//  TensorProperties.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/2/11.
//

#ifndef TensorProperties_hpp
#define TensorProperties_hpp

namespace otter {

class Tensor;

Tensor contiguous(const Tensor& self);

Tensor contiguous(const Tensor& self, MemoryFormat memory_format);


}

#endif /* TensorProperties_hpp */
