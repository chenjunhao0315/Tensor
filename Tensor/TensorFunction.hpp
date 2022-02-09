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
DECLEAR_META_STRUCTURE_OTHER(remainder, Tensor);
DECLEAR_META_STRUCTURE_OTHER(bitwise_and, Tensor);
DECLEAR_META_STRUCTURE_OTHER(bitwise_or, Tensor);
DECLEAR_META_STRUCTURE_OTHER(bitwise_xor, Tensor);

DECLEAR_META_STRUCTURE_SELF(bitwise_not, Tensor);
DECLEAR_META_STRUCTURE_SELF(neg, Tensor);
DECLEAR_META_STRUCTURE_SELF(abs, Tensor);
DECLEAR_META_STRUCTURE_SELF(sin, Tensor);
DECLEAR_META_STRUCTURE_SELF(cos, Tensor);
DECLEAR_META_STRUCTURE_SELF(tan, Tensor);
DECLEAR_META_STRUCTURE_SELF(exp, Tensor);

#define DEFINE_FINAL_OP(name, overload) \
struct structured_##name##_##overload##_functional : structured_##name##_##overload { \
    void set_output(int64_t output_idx, IntArrayRef sizes, IntArrayRef strides, TensorOptions options) override { \
        outputs_[output_idx] = create_out(sizes, strides, options); \
        structured_##name##_##overload::set_output(output_idx, sizes, strides, options); \
    } \
    const Tensor& maybe_get_output(int64_t output_idx) override { \
        return *outputs_[output_idx]; \
    } \
    std::array<otter::ExclusivelyOwned<Tensor>, 1> outputs_; \
}; \
struct structured_##name##_##overload##_out : structured_##name##_##overload { \
    structured_##name##_##overload##_out(Tensor& out0) : outputs_{ std::ref(out0) } {} \
    void set_output(int64_t output_idx, IntArrayRef sizes, IntArrayRef strides, TensorOptions options) override { \
        const auto& out = outputs_[output_idx].get(); \
        structured_##name##_##overload::set_output(output_idx, sizes, strides, options); \
    } \
    const Tensor& maybe_get_output(int64_t output_idx) override { \
        return outputs_[output_idx]; \
    } \
    std::array<std::reference_wrapper<Tensor>, 1> outputs_; \
};

DEFINE_FINAL_OP(add, Tensor);
DEFINE_FINAL_OP(sub, Tensor);
DEFINE_FINAL_OP(mul, Tensor);
DEFINE_FINAL_OP(div, Tensor);
DEFINE_FINAL_OP(remainder, Tensor);
DEFINE_FINAL_OP(bitwise_and, Tensor);
DEFINE_FINAL_OP(bitwise_or, Tensor);
DEFINE_FINAL_OP(bitwise_xor, Tensor);

DEFINE_FINAL_OP(bitwise_not, Tensor);
DEFINE_FINAL_OP(neg, Tensor);
DEFINE_FINAL_OP(abs, Tensor);
DEFINE_FINAL_OP(sin, Tensor);
DEFINE_FINAL_OP(cos, Tensor);
DEFINE_FINAL_OP(tan, Tensor);
DEFINE_FINAL_OP(exp, Tensor);


}

#endif /* TensorFunction_hpp */
