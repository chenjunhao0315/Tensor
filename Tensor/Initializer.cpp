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

InitializerFromDataReader::InitializerFromDataReader(const DataReader& dr_) : Initializer(), dr(dr_) {
    type = InitializerType::Otter;
}

InitializerFromDataReader::~InitializerFromDataReader() {
}

Tensor InitializerFromDataReader::load(IntArrayRef shape, int /*type*/) const {
    size_t nread = 0;
    void* refbuf = nullptr;
    Tensor result;
    
    const int64_t size = otter::multiply_integers(shape);
    
    nread = dr.reference(size * sizeof(float), &refbuf);
    
    if (nread == size * sizeof(float)) {
        result = otter::from_blob(refbuf, shape, otter::ScalarType::Float);
    } else {
        result = otter::empty(shape, otter::ScalarType::Float);
        nread = dr.read(result.raw_data(), size * sizeof(float));
        
        if (nread != size * sizeof(float)) {
            fprintf(stderr, "Load weight fail!\n");
            
            return Tensor();
        }
    }
    
    return result;
}

static size_t alignSize(size_t sz, int n) {
    return (sz + n - 1) & -n;
}

InitializerNcnnFromDataReader::InitializerNcnnFromDataReader(const DataReader& dr_) : Initializer(), dr(dr_) {
    type = InitializerType::Ncnn;
}

InitializerNcnnFromDataReader::~InitializerNcnnFromDataReader() {
}

Tensor InitializerNcnnFromDataReader::load(IntArrayRef shape, int type) const {
    const int64_t size = otter::multiply_integers(shape);
    
    if (type == 0) {
        size_t nread;
        
        union {
            struct {
                unsigned char f0;
                unsigned char f1;
                unsigned char f2;
                unsigned char f3;
            };
            unsigned int tag;
        } flag_struct;
        
        nread = dr.read(&flag_struct, sizeof(flag_struct));
        if (nread != sizeof(flag_struct)) {
            printf("ModelBin read flag_struct failed %zd\n", nread);
            return Tensor();
        }
        
        unsigned int flag = flag_struct.f0 + flag_struct.f1 + flag_struct.f2 + flag_struct.f3;
        
        if (flag_struct.tag == 0x01306B47) {
            // half-precision data
            // fp16
            OTTER_CHECK(false, "Unsupport fp16 data!");
        } else if (flag_struct.tag == 0x000D4B38) {
            // int8
            OTTER_CHECK(false, "Unsupport int8 data!");
        } else if (flag_struct.tag == 0x0002C056) {
            auto result = otter::empty(shape, ScalarType::Float);
            
            nread = dr.read(result.raw_data(), size * sizeof(float));
            if (nread != size * sizeof(float)) {
                printf("ModelBin read weight_data failed %zd\n", nread);
                return Tensor();
            }
            
            return result;
        }
        
        auto result = otter::empty(shape, ScalarType::Float);
        
        if (flag != 0) {
            // quantized data
            float quantization_value[256];
            nread = dr.read(quantization_value, 256 * sizeof(float));
            if (nread != 256 * sizeof(float)) {
                printf("ModelBin read quantization_value failed %zd\n", nread);
                return Tensor();
            }
            
            size_t align_weight_data_size = alignSize(size * sizeof(unsigned char), 4);
            std::vector<unsigned char> index_array;
            index_array.resize(align_weight_data_size);
            nread = dr.read(index_array.data(), align_weight_data_size);
            if (nread != align_weight_data_size) {
                printf("ModelBin read index_array failed %zd\n", nread);
                return Tensor();
            }
            
            float* ptr = result.data_ptr<float>();
            for (int i = 0; i < size; i++) {
                ptr[i] = quantization_value[index_array[i]];
            }
        } else if (flag_struct.f0 == 0) {
            // raw data
            nread = dr.read(result.raw_data(), size * sizeof(float));
            if (nread != size * sizeof(float)) {
                printf("ModelBin read weight_data failed %zd\n", nread);
                return Tensor();
            }
        }
        
        return result;
        
    } else if (type == 1) {
        auto result = otter::empty(shape, ScalarType::Float);
        
        size_t nread = dr.read(result.raw_data(), size * sizeof(float));
        if (nread != size * sizeof(float)) {
            printf("ModelBin read weight_data failed %zd\n", nread);
            return Tensor();
        }
        
        return result;
    } else {
        OTTER_CHECK(false, "Unsupport ncnn weight type!");
        
        return Tensor();
    }
    
    
    return Tensor();
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
    type = InitializerType::XavierNormal;
}

InitializerXavierNormal::~InitializerXavierNormal() {
}

Tensor InitializerXavierNormal::load(IntArrayRef shape, int /*type*/) const {
    Tensor result = otter::empty(shape, otter::ScalarType::Float);
    
    Fan fan(result);
    
    const auto std = gain * std::sqrt(2.0 / (fan.in + fan.out));
    return result.normal_(0, std);
}

InitializerXavierUniform::InitializerXavierUniform(double gain_) : gain(gain_) {
    type = InitializerType::XavierUniform;
}

InitializerXavierUniform::~InitializerXavierUniform() {
}

Tensor InitializerXavierUniform::load(IntArrayRef shape, int /*type*/) const {
    Tensor result = otter::empty(shape, otter::ScalarType::Float);
    
    Fan fan(result);
    
    const auto std = gain * std::sqrt(2.0 / (fan.in + fan.out));
    // Calculate uniform bounds from standard deviation with
    const auto a = std::sqrt(3.0) * std;
    return result.uniform_(-a, a);
}

}   // end namespace otter
