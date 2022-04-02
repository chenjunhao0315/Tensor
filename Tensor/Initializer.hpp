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

enum class InitializerType {
    Otter,
    Ncnn,
    XavierNormal,
    XavierUniform
};

class Initializer {
public:
    Initializer();
    
    virtual ~Initializer();
    
    virtual Tensor load(IntArrayRef shape, int type) const = 0;
    
public:
    InitializerType type;
};

class InitializerFromDataReader : public Initializer {
public:
    InitializerFromDataReader(const DataReader& dr_);
    virtual ~InitializerFromDataReader();
    
    virtual Tensor load(IntArrayRef shape, int type) const;
private:
    const DataReader& dr;
};

class InitializerNcnnFromDataReader : public Initializer {
public:
    InitializerNcnnFromDataReader(const DataReader& dr_);
    virtual ~InitializerNcnnFromDataReader();
    
    virtual Tensor load(IntArrayRef shape, int type) const;
private:
    const DataReader& dr;
};

class InitializerXavierNormal : public Initializer {
public:
    InitializerXavierNormal(double gain_);
    virtual ~InitializerXavierNormal();
    
    virtual Tensor load(IntArrayRef shape, int type) const;
private:
    double gain;
};

class InitializerXavierUniform : public Initializer {
public:
    InitializerXavierUniform(double gain_);
    virtual ~InitializerXavierUniform();
    
    virtual Tensor load(IntArrayRef shape, int type) const;
private:
    double gain;
};

}

#endif /* Initializer_hpp */
