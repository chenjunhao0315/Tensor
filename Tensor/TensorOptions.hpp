//
//  TensorOptionss.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/1/29.
//

#ifndef TensorOptionss_hpp
#define TensorOptionss_hpp

#include "Allocator.hpp"
#include "DType.hpp"

namespace otter {

struct TensorOptions {
    TensorOptions() : has_device_(false), has_data_type_(false), has_required_grad_(false) {}
    
    TensorOptions(ScalarType dtype) : TensorOptions() {
        this->set_dtype(dtype);
    }
    
    Device device() const noexcept {
        return device_;
    }
    
    bool has_device() const noexcept {
        return has_device_;
    }
    
    TypeMeta dtype() const noexcept {
        return data_type_;
    }
    
    bool has_dtype() const noexcept {
        return has_data_type_;
    }
    
    bool required_grad() const noexcept {
        return has_required_grad_ ? required_grad_ : false;
    }
    
    TensorOptions device(Device device) const noexcept {
        TensorOptions t = *this;
        t.set_device(device);
        return t;
    }
    
    TensorOptions dtype(TypeMeta data_type) const noexcept {
        TensorOptions t = *this;
        t.set_dtype(data_type);
        return t;
    }
    
    template <typename T>
    TensorOptions& dtype() {
        data_type_ = TypeMeta::Make<T>();
        has_data_type_ = true;
        return *this;
    }
    
    TensorOptions required_grad(bool required) const noexcept {
        TensorOptions t = *this;
        t.set_required_grad(required);
        return t;
    }
    
private:
    void set_device(Device device) & noexcept {
        device_ = device;
        has_device_ = true;
    }
    
    void set_dtype(TypeMeta data_type) & noexcept {
        data_type_ = data_type;
        has_data_type_ = true;
    }
    
    void set_dtype(ScalarType dtype) & noexcept {
        data_type_ = scalarTypeToTypeMeta(dtype);
        has_data_type_ = true;
    }
    
    void set_required_grad(bool required_grad) & noexcept {
        required_grad_ = required_grad;
        has_required_grad_ = true;
    }
    
    
    Device device_ = Device::CPU;
    TypeMeta data_type_ = TypeMeta::Make<float>();
    
    bool required_grad_ : 1;
    
    bool has_device_ : 1;
    bool has_data_type_ : 1;
    bool has_required_grad_ : 1;
};

}   // end namespace otter

#endif /* TensorOptionss_hpp */
