//
//  Initializer.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/3/8.
//

#include "Initializer.hpp"
#include "TensorMaker.hpp"
#include "TensorFactory.hpp"
#include "Accumulator.hpp"

namespace otter {

Initializer::Initializer() {
}

Initializer::~Initializer() {
}

InitializerFromDataReader::InitializerFromDataReader(const DataReader& dr) : Initializer(), dr_(dr) {
}

InitializerFromDataReader::~InitializerFromDataReader() {
}

Tensor InitializerFromDataReader::load(IntArrayRef shape) const {
    int64_t nread = 0;
    void* refbuf = nullptr;
    Tensor result;
    
    const int64_t size = otter::multiply_integers(shape);
    
    nread = dr_.reference(size * sizeof(float), &refbuf);
    
    if (nread == size * sizeof(float)) {
        result = otter::from_blob(refbuf, shape, otter::ScalarType::Float);
    } else {
        result = otter::empty(shape, otter::ScalarType::Float);
        nread = dr_.read(result.raw_data(), size * sizeof(float));
        
        if (nread != size * sizeof(float)) {
            fprintf(stderr, "Load weight fail!\n");
            
            return Tensor();
        }
    }
    
    return result;
}

}
