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
void check_inplace(const Tensor &self, IntArrayRef sizes, const TensorOptions &options);

#define DECLARE_META_STRUCTURE_DUAL_NONE(name) \
struct structured_##name : public TensorIterator {  \
    void meta(const Tensor& self, const Tensor& other);   \
}

#define DECLARE_META_STRUCTURE_TRI_DUAL(name) \
struct structured_##name : public TensorIterator { \
    void meta(const Tensor& self, const Tensor& mat1, const Tensor& mat2, const Scalar& beta, const Scalar& alpha); \
}

#define DECLARE_META_STRUCTURE_SIN_SIN(name) \
struct structured_##name : public TensorIterator { \
    void meta(const Tensor& self, const Scalar& alpha); \
}

#define DECLARE_META_STRUCTURE_SIN_DUAL(name) \
struct structured_##name : public TensorIterator { \
    void meta(const Tensor& self, const Scalar& alpha, const Scalar& beta); \
}

#define DECLARE_META_STRUCTURE_SELF_OVERLOAD(name, overload)    \
struct structured_##name##_##overload : public TensorIterator {   \
    void meta(const Tensor& self);   \
}

#define DECLARE_META_STRUCTURE_OTHER_OVERLOAD(name, overload)    \
struct structured_##name##_##overload : public TensorIterator  {    \
    void meta(const Tensor& self, const Tensor& other);   \
}

#define DECLARE_META_STRUCTURE_OTHER_WITH_SCALAR_OVERLOAD(name, overload)    \
struct structured_##name##_##overload : public TensorIterator  {    \
    void meta(const Tensor& self, const Tensor& other, const Scalar& value);   \
}

#define DEFINE_META_FUNCTION(name)  \
void structured_##name::meta

#define DEFINE_IMPL_FUNCTION(name)  \
void structured_##name::impl

#define DEFINE_META_FUNCTION_OVERLOAD(name, overload)  \
void structured_##name##_##overload::meta

DECLARE_META_STRUCTURE_OTHER_WITH_SCALAR_OVERLOAD(add, Tensor);
DECLARE_META_STRUCTURE_OTHER_WITH_SCALAR_OVERLOAD(sub, Tensor);
DECLARE_META_STRUCTURE_OTHER_OVERLOAD(mul, Tensor);
DECLARE_META_STRUCTURE_OTHER_OVERLOAD(div, Tensor);
DECLARE_META_STRUCTURE_OTHER_OVERLOAD(remainder, Tensor);
DECLARE_META_STRUCTURE_OTHER_OVERLOAD(bitwise_and, Tensor);
DECLARE_META_STRUCTURE_OTHER_OVERLOAD(bitwise_or, Tensor);
DECLARE_META_STRUCTURE_OTHER_OVERLOAD(bitwise_xor, Tensor);

DECLARE_META_STRUCTURE_SELF_OVERLOAD(bitwise_not, Tensor);
DECLARE_META_STRUCTURE_SELF_OVERLOAD(neg, Tensor);
DECLARE_META_STRUCTURE_SELF_OVERLOAD(abs, Tensor);
DECLARE_META_STRUCTURE_SELF_OVERLOAD(sin, Tensor);
DECLARE_META_STRUCTURE_SELF_OVERLOAD(cos, Tensor);
DECLARE_META_STRUCTURE_SELF_OVERLOAD(tan, Tensor);
DECLARE_META_STRUCTURE_SELF_OVERLOAD(exp, Tensor);
DECLARE_META_STRUCTURE_SELF_OVERLOAD(sqrt, Tensor);

DECLARE_META_STRUCTURE_TRI_DUAL(addmm);
DECLARE_META_STRUCTURE_DUAL_NONE(mm);

DECLARE_META_STRUCTURE_SIN_SIN(leaky_relu);
DECLARE_META_STRUCTURE_SIN_DUAL(threshold);

struct structured_max_pool2d_with_indices : public TensorIterator {
    void meta(const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode);
};

struct structured_upsample_nearest2d : public TensorIterator {
    void meta(const Tensor & self, IntArrayRef output_size, double scales_h, double scales_w);
};

struct structured_upsample_bilinear2d : public TensorIterator {
    void meta(const Tensor & self, IntArrayRef output_size, bool align_corners, double scales_h, double scales_w);
};

struct structured_eq_Scalar : public TensorIterator {
    void meta(const Tensor & self, const Scalar & other);
};

struct structured_eq_Tensor : public TensorIterator {
    void meta(const Tensor & self, const Tensor & other);
};

struct structured_ne_Scalar : public TensorIterator {
    void meta(const Tensor & self, const Scalar & other);
};

struct structured_ne_Tensor : public TensorIterator {
    void meta(const Tensor & self, const Tensor & other);
};

struct structured_ge_Scalar : public TensorIterator {
    void meta(const Tensor & self, const Scalar & other);
};

struct structured_ge_Tensor : public TensorIterator {
    void meta(const Tensor & self, const Tensor & other);
};

struct structured_gt_Scalar : public TensorIterator {
    void meta(const Tensor & self, const Scalar & other);
};

struct structured_gt_Tensor : public TensorIterator {
    void meta(const Tensor & self, const Tensor & other);
};

struct structured_le_Scalar : public TensorIterator {
    void meta(const Tensor & self, const Scalar & other);
};

struct structured_le_Tensor : public TensorIterator {
    void meta(const Tensor & self, const Tensor & other);
};

struct structured_lt_Scalar : public TensorIterator {
    void meta(const Tensor & self, const Scalar & other);
};

struct structured_lt_Tensor : public TensorIterator {
    void meta(const Tensor & self, const Tensor & other);
};

struct structured_clamp : public TensorIterator {
    void meta(const Tensor & self, Scalar min, Scalar max);
};

#define DEFINE_FINAL_OP_AFTER(name) \
struct structured_##name##_functional : structured_##name { \
    void set_output(int64_t output_idx, IntArrayRef sizes, IntArrayRef strides, TensorOptions options) override { \
        outputs_[output_idx] = create_out(sizes, strides, options); \
        structured_##name::set_output(output_idx, sizes, strides, options); \
    } \
    const Tensor& maybe_get_output(int64_t output_idx) override { \
        return *outputs_[output_idx]; \
    } \
    std::array<ExclusivelyOwned<Tensor>, 1> outputs_; \
}; \
struct structured_##name##_out : structured_##name { \
    structured_##name##_out(Tensor& out0) : outputs_{ std::ref(out0) } {} \
    void set_output(int64_t output_idx, IntArrayRef sizes, IntArrayRef strides, TensorOptions options) override { \
        const auto& out = outputs_[output_idx].get(); \
        resize_out(out, sizes, strides, options); \
        structured_##name::set_output(output_idx, sizes, strides, options); \
    } \
    const Tensor& maybe_get_output(int64_t output_idx) override { \
        return outputs_[output_idx]; \
    } \
    std::array<std::reference_wrapper<Tensor>, 1> outputs_; \
}; \
struct structured_##name##_inplace : structured_##name { \
    structured_##name##_inplace(Tensor& out0) : outputs_{ std::ref(out0) } {} \
    void set_output(int64_t output_idx, IntArrayRef sizes, IntArrayRef strides, TensorOptions options) override { \
        const auto& out = outputs_[output_idx].get(); \
        resize_out(out, sizes, strides, options); \
        structured_##name::set_output(output_idx, sizes, strides, options); \
    } \
    const Tensor& maybe_get_output(int64_t output_idx) override { \
        return outputs_[output_idx]; \
    } \
    std::array<std::reference_wrapper<Tensor>, 1> outputs_; \
};

struct structured_add_out : structured_add_Tensor {
    void impl(const Tensor & self, const Tensor & other, const Scalar & alpha, const Tensor & out);
};

struct structured_sub_out : structured_sub_Tensor {
    void impl(const Tensor & self, const Tensor & other, const Scalar & alpha, const Tensor & out);
};

struct structured_mul_out : structured_mul_Tensor {
    void impl(const Tensor & self, const Tensor & other, const Tensor & out);
};

struct structured_div_out : structured_div_Tensor {
    void impl(const Tensor & self, const Tensor & other, const Tensor & out);
};

struct structured_remainder_out : structured_remainder_Tensor {
    void impl(const Tensor & self, const Tensor & other, const Tensor & out);
};

struct structured_bitwise_and_out : structured_bitwise_and_Tensor {
    void impl(const Tensor & self, const Tensor & other, const Tensor & out);
};

struct structured_bitwise_or_out : structured_bitwise_or_Tensor {
    void impl(const Tensor & self, const Tensor & other, const Tensor & out);
};

struct structured_bitwise_xor_out : structured_bitwise_xor_Tensor {
    void impl(const Tensor & self, const Tensor & other, const Tensor & out);
};

struct structured_bitwise_not_out : structured_bitwise_not_Tensor {
    void impl(const Tensor & self, const Tensor & out);
};

struct structured_neg_out : structured_neg_Tensor {
    void impl(const Tensor & self, const Tensor & out);
};

struct structured_abs_out : structured_abs_Tensor {
    void impl(const Tensor & self, const Tensor & out);
};

struct structured_sin_out : structured_sin_Tensor {
    void impl(const Tensor & self, const Tensor & out);
};

struct structured_cos_out : structured_cos_Tensor {
    void impl(const Tensor & self, const Tensor & out);
};

struct structured_tan_out : structured_tan_Tensor {
    void impl(const Tensor & self, const Tensor & out);
};

struct structured_exp_out : structured_exp_Tensor {
    void impl(const Tensor & self, const Tensor & out);
};

struct structured_sqrt_out : structured_sqrt_Tensor {
    void impl(const Tensor & self, const Tensor & out);
};

struct structured_addmm_out_cpu : structured_addmm {
    void impl(const Tensor & self, const Tensor & mat1, const Tensor & mat2, const Scalar & beta, const Scalar & alpha, const Tensor & out);
};

struct structured_mm_out_cpu : structured_mm {
    void impl(const Tensor & self, const Tensor & other, const Tensor & out);
};

struct structured_leaky_relu_out : structured_leaky_relu {
    void impl(const Tensor & self, const Scalar & alpha, const Tensor & out);
};

struct structured_max_pool2d_with_indices_out_cpu : public structured_max_pool2d_with_indices {
    void impl(const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode, const Tensor & out, const Tensor & indices);
};

struct structured_upsample_nearest2d_out_cpu : public structured_upsample_nearest2d {
    void impl(const Tensor & self, IntArrayRef output_size, double scales_h, double scales_w, const Tensor & out);
};

struct structured_upsample_bilinear2d_out_cpu : public structured_upsample_bilinear2d {
    void impl(const Tensor & self, IntArrayRef output_size, bool align_corners, double scales_h, double scales_w, const Tensor & out);
};

struct structured_threshold_out : public structured_threshold {
    void impl(const Tensor & self, const Scalar & threshold, const Scalar & value, const Tensor & out);
};

struct structured_eq_Scalar_out : public structured_eq_Scalar {
    void impl(const Tensor & self, const Scalar & other, const Tensor & out);
};

struct structured_eq_Tensor_out : public structured_eq_Tensor {
    void impl(const Tensor & self, const Tensor & other, const Tensor & out);
};

struct structured_ne_Scalar_out : public structured_ne_Scalar {
    void impl(const Tensor & self, const Scalar & other, const Tensor & out);
};

struct structured_ne_Tensor_out : public structured_ne_Tensor {
    void impl(const Tensor & self, const Tensor & other, const Tensor & out);
};

struct structured_ge_Scalar_out : public structured_ge_Scalar {
    void impl(const Tensor & self, const Scalar & other, const Tensor & out);
};

struct structured_ge_Tensor_out : public structured_ge_Tensor {
    void impl(const Tensor & self, const Tensor & other, const Tensor & out);
};

struct structured_gt_Scalar_out : public structured_gt_Scalar {
    void impl(const Tensor & self, const Scalar & other, const Tensor & out);
};

struct structured_gt_Tensor_out : public structured_gt_Tensor {
    void impl(const Tensor & self, const Tensor & other, const Tensor & out);
};

struct structured_le_Scalar_out : public structured_le_Scalar {
    void impl(const Tensor & self, const Scalar & other, const Tensor & out);
};

struct structured_le_Tensor_out : public structured_le_Tensor {
    void impl(const Tensor & self, const Tensor & other, const Tensor & out);
};

struct structured_lt_Scalar_out : public structured_lt_Scalar {
    void impl(const Tensor & self, const Scalar & other, const Tensor & out);
};

struct structured_lt_Tensor_out : public structured_lt_Tensor {
    void impl(const Tensor & self, const Tensor & other, const Tensor & out);
};

struct structured_clamp_out : public structured_clamp {
    void impl(const Tensor & self, Scalar min, Scalar max, const Tensor & out);
};

namespace native {

Tensor add(const Tensor & self, const Tensor & other, const Scalar & alpha);
Tensor & add_out(Tensor & out, const Tensor & self, const Tensor & other, const Scalar & alpha);
Tensor & add_(Tensor & self, const Tensor & other, const Scalar & alpha);

Tensor sub(const Tensor & self, const Tensor & other, const Scalar & alpha);
Tensor & sub_out(Tensor & out, const Tensor & self, const Tensor & other, const Scalar & alpha);
Tensor & sub_(Tensor & self, const Tensor & other, const Scalar & alpha);

Tensor mul(const Tensor & self, const Tensor & other);
Tensor & mul_out(Tensor & out, const Tensor & self, const Tensor & other);
Tensor & mul_(Tensor & self, const Tensor & other);

Tensor div(const Tensor & self, const Tensor & other);
Tensor & div_out(Tensor & out, const Tensor & self, const Tensor & other);
Tensor & div_(Tensor & self, const Tensor & other);

Tensor remainder(const Tensor & self, const Tensor & other);
Tensor & remainder_out(Tensor & out, const Tensor & self, const Tensor & other);
Tensor & remainder_(Tensor & self, const Tensor & other);

Tensor bitwise_and(const Tensor & self, const Tensor & other);
Tensor & bitwise_and_out(Tensor & out, const Tensor & self, const Tensor & other);
Tensor & bitwise_and_(Tensor & self, const Tensor & other);

Tensor bitwise_or(const Tensor & self, const Tensor & other);
Tensor & bitwise_or_out(Tensor & out, const Tensor & self, const Tensor & other);
Tensor & bitwise_or_(Tensor & self, const Tensor & other);

Tensor bitwise_xor(const Tensor & self, const Tensor & other);
Tensor & bitwise_xor_out(Tensor & out, const Tensor & self, const Tensor & other);
Tensor & bitwise_xor_(Tensor & self, const Tensor & other);

Tensor neg(const Tensor & self);
Tensor & neg_out(Tensor & out, const Tensor & self);
Tensor & neg_(Tensor & self);

Tensor bitwise_not(const Tensor & self);
Tensor & bitwise_not_out(Tensor & out, const Tensor & self);
Tensor & bitwise_not_(Tensor & self);

Tensor abs(const Tensor & self);
Tensor & abs_out(Tensor & out, const Tensor & self);
Tensor & abs_(Tensor & self);

Tensor sin(const Tensor & self);
Tensor & sin_out(Tensor & out, const Tensor & self);
Tensor & sin_(Tensor & self);

Tensor cos(const Tensor & self);
Tensor & cos_out(Tensor & out, const Tensor & self);
Tensor & cos_(Tensor & self);

Tensor tan(const Tensor & self);
Tensor & tan_out(Tensor & out, const Tensor & self);
Tensor & tan_(Tensor & self);

Tensor exp(const Tensor & self);
Tensor & exp_out(Tensor & out, const Tensor & self);
Tensor & exp_(Tensor & self);

Tensor sqrt(const Tensor & self);
Tensor & sqrt_out(Tensor & out, const Tensor & self);
Tensor & sqrt_(Tensor & self);

Tensor addmm(const Tensor & self, const Tensor & mat1, const Tensor & mat2, const Scalar & beta, const Scalar & alpha);
Tensor & addmm_out(Tensor & out, const Tensor & self, const Tensor & mat1, const Tensor & mat2, const Scalar & beta, const Scalar & alpha);
Tensor & addmm_(Tensor & self, const Tensor & mat1, const Tensor & mat2, const Scalar & beta, const Scalar & alpha);

Tensor mm(const Tensor & self, const Tensor & other);
Tensor & mm_out(Tensor & out, const Tensor & self, const Tensor & other);
Tensor & mm_(Tensor & self, const Tensor & other);

Tensor leaky_relu(const Tensor & self, const Scalar& negative_slope);
Tensor & leaky_relu_out(Tensor & out, Tensor & self, const Scalar & negative_slope);
Tensor & leaky_relu_(Tensor & self, const Scalar & negative_slope);

std::tuple<Tensor, Tensor> max_pool2d_with_indices(const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode);
std::tuple<Tensor&, Tensor&> max_pool2d_with_indices_out(Tensor & out, Tensor & indices, const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode);

Tensor upsample_nearest2d(const Tensor & self, IntArrayRef output_size, double scales_h, double scales_w);
Tensor & upsample_nearest2d_out(Tensor & out, const Tensor & self, IntArrayRef output_size, double scales_h, double scales_w);

Tensor upsample_bilinear2d(const Tensor & self, IntArrayRef output_size, bool align_corners, double scales_h, double scales_w);
Tensor & upsample_bilinear2d_out(Tensor & out, const Tensor & self, IntArrayRef output_size, bool align_corners, double scales_h, double scales_w);

Tensor threshold(const Tensor & self, const Scalar & threshold, const Scalar & value);
Tensor & threshold_out(Tensor & out, const Tensor & self, const Scalar & threshold, const Scalar & value);
Tensor & threshold_(Tensor & self, const Scalar & threshold, const Scalar & value);

Tensor eq(const Tensor & self, const Scalar & other);
Tensor & eq_out(Tensor & out, const Tensor & self, const Scalar & other);
Tensor & eq_outf(const Tensor & self, const Scalar & other, Tensor & out);
Tensor & eq_(Tensor & self, const Scalar & other);
Tensor eq(const Tensor & self, const Tensor & other);
Tensor & eq_out(Tensor & out, const Tensor & self, const Tensor & other);
Tensor & eq_outf(const Tensor & self, const Tensor & other, Tensor & out);
Tensor & eq_(Tensor & self, const Tensor & other);

Tensor ne(const Tensor & self, const Scalar & other);
Tensor & ne_out(Tensor & out, const Tensor & self, const Scalar & other);
Tensor & ne_outf(const Tensor & self, const Scalar & other, Tensor & out);
Tensor & ne_(Tensor & self, const Scalar & other);
Tensor ne(const Tensor & self, const Tensor & other);
Tensor & ne_out(Tensor & out, const Tensor & self, const Tensor & other);
Tensor & ne_outf(const Tensor & self, const Tensor & other, Tensor & out);
Tensor & ne_(Tensor & self, const Tensor & other);

Tensor ge(const Tensor & self, const Scalar & other);
Tensor & ge_out(Tensor & out, const Tensor & self, const Scalar & other);
Tensor & ge_outf(const Tensor & self, const Scalar & other, Tensor & out);
Tensor & ge_(Tensor & self, const Scalar & other);
Tensor ge(const Tensor & self, const Tensor & other);
Tensor & ge_out(Tensor & out, const Tensor & self, const Tensor & other);
Tensor & ge_outf(const Tensor & self, const Tensor & other, Tensor & out);
Tensor & ge_(Tensor & self, const Tensor & other);

Tensor le(const Tensor & self, const Scalar & other);
Tensor & le_out(Tensor & out, const Tensor & self, const Scalar & other);
Tensor & le_outf(const Tensor & self, const Scalar & other, Tensor & out);
Tensor & le_(Tensor & self, const Scalar & other);
Tensor le(const Tensor & self, const Tensor & other);
Tensor & le_out(Tensor & out, const Tensor & self, const Tensor & other);
Tensor & le_outf(const Tensor & self, const Tensor & other, Tensor & out);
Tensor & le_(Tensor & self, const Tensor & other);

Tensor gt(const Tensor & self, const Scalar & other);
Tensor & gt_out(Tensor & out, const Tensor & self, const Scalar & other);
Tensor & gt_outf(const Tensor & self, const Scalar & other, Tensor & out);
Tensor & gt_(Tensor & self, const Scalar & other);
Tensor gt(const Tensor & self, const Tensor & other);
Tensor & gt_out(Tensor & out, const Tensor & self, const Tensor & other);
Tensor & gt_outf(const Tensor & self, const Tensor & other, Tensor & out);
Tensor & gt_(Tensor & self, const Tensor & other);

Tensor lt(const Tensor & self, const Scalar & other);
Tensor & lt_out(Tensor & out, const Tensor & self, const Scalar & other);
Tensor & lt_outf(const Tensor & self, const Scalar & other, Tensor & out);
Tensor & lt_(Tensor & self, const Scalar & other);
Tensor lt(const Tensor & self, const Tensor & other);
Tensor & lt_out(Tensor & out, const Tensor & self, const Tensor & other);
Tensor & lt_outf(const Tensor & self, const Tensor & other, Tensor & out);
Tensor & lt_(Tensor & self, const Tensor & other);

}   // end namespace native

}   // end namespace otter

#endif /* TensorFunction_hpp */
