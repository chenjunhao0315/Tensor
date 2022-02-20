//
//  Blob.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/2/19.
//

#ifndef Blob_hpp
#define Blob_hpp

#include "Tensor.hpp"

namespace otter {

class Blob {
public:
    Blob();
    
public:
    int producer;
    int consumer;
    std::string name;
    Tensor shape;
};

}

#endif /* Blob_hpp */
