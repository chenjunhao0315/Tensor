//
//  TensorCopyKernel.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/2/10.
//

#ifndef TensorCopyKernel_hpp
#define TensorCopyKernel_hpp

class TensorIterator;

namespace otter {

void copy_kernel(TensorIterator& iter, bool non_blocking);

}

#endif /* TensorCopyKernel_hpp */
