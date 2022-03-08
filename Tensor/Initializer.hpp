//
//  Initializer.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/3/8.
//

#ifndef Initializer_hpp
#define Initializer_hpp

#include "Tensor.hpp"
#include "DataReader.hpp"

namespace otter {

class Initializer {
public:
    Initializer();
    
    virtual ~Initializer();
    
    virtual Tensor load(IntArrayRef shape) const = 0;
};

class InitializerFromDataReader : public Initializer {
public:
    InitializerFromDataReader(const DataReader& dr);
    virtual ~InitializerFromDataReader();
    
    virtual Tensor load(IntArrayRef shape) const;
private:
    const DataReader& dr_;
};

}

#endif /* Initializer_hpp */
