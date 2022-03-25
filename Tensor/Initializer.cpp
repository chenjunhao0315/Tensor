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

namespace {
struct Fan {
    explicit Fan(Tensor& tensor) {
        const auto dimensions = tensor.dim();
        OTTER_CHECK(dimensions >= 2,
                    "Fan in and fan out can not be computed for tensor with fewer than 2 dimensions");
        
        if (dimensions == 2) {
            in = tensor.size(1);
            out = tensor.size(0);
        } else {
            in = tensor.size(1) * tensor[0][0].numel();
            out = tensor.size(0) * tensor[0][0].numel();
        }
    }
    
    int64_t in;
    int64_t out;
};
}   // end namespace

InitializerXavierNormal::InitializerXavierNormal(double gain_) : gain(gain_) {
}

InitializerXavierNormal::~InitializerXavierNormal() {
}

Tensor InitializerXavierNormal::load(IntArrayRef shape) const {
    Tensor result = otter::empty(shape, otter::ScalarType::Float);
    
    Fan fan(result);
    
    const auto std = gain * std::sqrt(2.0 / (fan.in + fan.out));
    return result.normal_(0, std);
}

InitializerXavierUniform::InitializerXavierUniform(double gain_) : gain(gain_) {
}

InitializerXavierUniform::~InitializerXavierUniform() {
}

Tensor InitializerXavierUniform::load(IntArrayRef shape) const {
    Tensor result = otter::empty(shape, otter::ScalarType::Float);
    
    Fan fan(result);
    
    const auto std = gain * std::sqrt(2.0 / (fan.in + fan.out));
    // Calculate uniform bounds from standard deviation with
    const auto a = std::sqrt(3.0) * std;
    return result.uniform_(-a, a);
}

}   // end namespace otter
