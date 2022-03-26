//
//  CPUAllocator.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/3/27.
//

#ifndef CPUAllocator_hpp
#define CPUAllocator_hpp

#include "Allocator.hpp"

namespace otter {

Allocator* GetCPUAllocator();
Allocator* GetAllocator(Device device);

Allocator* GetDefaultCPUAllocator();
Allocator* GetDefaultMobileCPUAllocator();

}

#endif /* CPUAllocator_hpp */
