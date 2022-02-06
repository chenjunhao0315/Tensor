//
//  TensorFunction.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/1/31.
//

#ifndef TensorFunction_hpp
#define TensorFunction_hpp

#include "ExclusivelyOwned.hpp"
#include "TensorIterator.hpp"

namespace otter {

Tensor create_out(IntArrayRef sizes, IntArrayRef strides, TensorOptions option);
void resize_out(const Tensor &out, IntArrayRef sizes, IntArrayRef strides, const TensorOptions &options);

DECLEAR_META_STRUCTURE_OTHER_WITH_SCALAR(add, Tensor);
DECLEAR_META_STRUCTURE_OTHER_WITH_SCALAR(sub, Tensor);
DECLEAR_META_STRUCTURE_OTHER(mul, Tensor);
DECLEAR_META_STRUCTURE_OTHER(div, Tensor);

struct structured_add_Tensor_functional : structured_add_Tensor {
    void set_output(int64_t output_idx, IntArrayRef sizes, IntArrayRef strides, TensorOptions options) override {
        outputs_[output_idx] = create_out(sizes, strides, options);
        
        structured_add_Tensor::set_output(output_idx, sizes, strides, options);
    }
    const Tensor& maybe_get_output(int64_t output_idx) override {
        return *outputs_[output_idx];
    }
    std::array<otter::ExclusivelyOwned<Tensor>, 1> outputs_;
};

struct structured_add_Tensor_out : structured_add_Tensor {
    structured_add_Tensor_out(Tensor& out0) : outputs_{ std::ref(out0) } {}
    
    void set_output(int64_t output_idx, IntArrayRef sizes, IntArrayRef strides, TensorOptions options) override {
        const auto& out = outputs_[output_idx].get();
        // TODO: resize out
        
        structured_add_Tensor::set_output(output_idx, sizes, strides, options);
    }
    const Tensor& maybe_get_output(int64_t output_idx) override {
        return outputs_[output_idx];
    }
    std::array<std::reference_wrapper<Tensor>, 1> outputs_;
};

struct structured_sub_Tensor_functional : structured_sub_Tensor {
    void set_output(int64_t output_idx, IntArrayRef sizes, IntArrayRef strides, TensorOptions options) override {
        outputs_[output_idx] = create_out(sizes, strides, options);
        
        structured_sub_Tensor::set_output(output_idx, sizes, strides, options);
    }
    const Tensor& maybe_get_output(int64_t output_idx) override {
        return *outputs_[output_idx];
    }
    std::array<otter::ExclusivelyOwned<Tensor>, 1> outputs_;
};

struct structured_sub_Tensor_out : structured_sub_Tensor {
    structured_sub_Tensor_out(Tensor& out0) : outputs_{ std::ref(out0) } {}
    
    void set_output(int64_t output_idx, IntArrayRef sizes, IntArrayRef strides, TensorOptions options) override {
        const auto& out = outputs_[output_idx].get();
        // TODO: resize out
        
        structured_sub_Tensor::set_output(output_idx, sizes, strides, options);
    }
    const Tensor& maybe_get_output(int64_t output_idx) override {
        return outputs_[output_idx];
    }
    std::array<std::reference_wrapper<Tensor>, 1> outputs_;
};

struct structured_mul_Tensor_functional : structured_mul_Tensor {
    void set_output(int64_t output_idx, IntArrayRef sizes, IntArrayRef strides, TensorOptions options) override {
        outputs_[output_idx] = create_out(sizes, strides, options);
        
        structured_mul_Tensor::set_output(output_idx, sizes, strides, options);
    }
    const Tensor& maybe_get_output(int64_t output_idx) override {
        return *outputs_[output_idx];
    }
    std::array<otter::ExclusivelyOwned<Tensor>, 1> outputs_;
};

struct structured_mul_Tensor_out : structured_mul_Tensor {
    structured_mul_Tensor_out(Tensor& out0) : outputs_{ std::ref(out0) } {}
    
    void set_output(int64_t output_idx, IntArrayRef sizes, IntArrayRef strides, TensorOptions options) override {
        const auto& out = outputs_[output_idx].get();
        // TODO: resize out
        
        structured_mul_Tensor::set_output(output_idx, sizes, strides, options);
    }
    const Tensor& maybe_get_output(int64_t output_idx) override {
        return outputs_[output_idx];
    }
    std::array<std::reference_wrapper<Tensor>, 1> outputs_;
};

struct structured_div_Tensor_functional : structured_div_Tensor {
    void set_output(int64_t output_idx, IntArrayRef sizes, IntArrayRef strides, TensorOptions options) override {
        outputs_[output_idx] = create_out(sizes, strides, options);
        
        structured_div_Tensor::set_output(output_idx, sizes, strides, options);
    }
    const Tensor& maybe_get_output(int64_t output_idx) override {
        return *outputs_[output_idx];
    }
    std::array<otter::ExclusivelyOwned<Tensor>, 1> outputs_;
};

struct structured_div_Tensor_out : structured_div_Tensor {
    structured_div_Tensor_out(Tensor& out0) : outputs_{ std::ref(out0) } {}
    
    void set_output(int64_t output_idx, IntArrayRef sizes, IntArrayRef strides, TensorOptions options) override {
        const auto& out = outputs_[output_idx].get();
        // TODO: resize out
        
        structured_div_Tensor::set_output(output_idx, sizes, strides, options);
    }
    const Tensor& maybe_get_output(int64_t output_idx) override {
        return outputs_[output_idx];
    }
    std::array<std::reference_wrapper<Tensor>, 1> outputs_;
};


}

#endif /* TensorFunction_hpp */
