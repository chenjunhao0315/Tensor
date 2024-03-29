//
//  TensorFunction.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/1/31.
//

#ifndef TensorFunction_hpp
#define TensorFunction_hpp

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
DECLARE_META_STRUCTURE_SELF_OVERLOAD(sigmoid, Tensor);

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

struct structured_softmax : public TensorIterator {
    void meta(const Tensor & self, int64_t dim, bool half_to_float);
};

struct structured_topk : public TensorIterator {
    void meta(const Tensor & self, int64_t k, int64_t dim, bool largest, bool sorted);
};

struct structured_sort_stable : public TensorIterator {
    void meta(const Tensor & self, bool stable, int64_t dim, bool descending);
};

struct structured_scatter_src : public TensorIterator {
    void meta(const Tensor & self, int64_t dim, const Tensor & index, const Tensor & src);
};

struct structured_scatter_value : public TensorIterator {
    void meta(const Tensor & self, int64_t dim, const Tensor & index, const Scalar & value);
};

struct structured_scatter_reduce : public TensorIterator {
    void meta(const Tensor & self, int64_t dim, const Tensor & index, const Tensor & src, int64_t reduce);
};

struct structured_scatter_value_reduce : public TensorIterator {
    void meta(const Tensor & self, int64_t dim, const Tensor & index, const Scalar & value, int64_t reduce);
};

struct structured_gather : public TensorIterator {
    void meta(const Tensor & self, int64_t dim, const Tensor & index, bool sparse_grad);
};

//struct structured_scatter_reduce_two : public TensorIterator {
//    void meta(const Tensor & self, int64_t dim, const Tensor & index, const Tensor & src, int64_t reduce, bool include_self);
//};
//
//struct structured_scatter_add : public TensorIterator {
//    void meta(const Tensor & self, int64_t dim, const Tensor & index, const Tensor & src);
//};

struct structured_bmm : public TensorIterator {
    void meta(const Tensor & self, const Tensor & mat2);
};

struct structured_baddbmm : public TensorIterator {
    void meta(const Tensor & self, const Tensor & batch1, const Tensor & batch2, const Scalar & beta, const Scalar & alpha);
};

struct structured_sum_dim_IntList : public TensorIterator {
    void meta(const Tensor & self, IntArrayRef dim, bool keepdim, ScalarType dtype);
};

struct structured_index_Tensor : public TensorIterator {
    
    template <bool SIZES = false, bool STRIDES = false>
    struct precompute_out {
        precompute_out<true, STRIDES> set_sizes(DimVector value) {
            static_assert(SIZES == false, "sizes already set");
            precompute_out<true, STRIDES> ret;
            ret.sizes = value;
            ret.strides = this->strides;
            return ret;
        }
                
        precompute_out<SIZES, true> set_strides(DimVector value) {
            static_assert(STRIDES == false, "strides already set");
            precompute_out<SIZES, true> ret;
            ret.sizes = this->sizes;
            ret.strides = value;
            return ret;
        }
                
        DimVector sizes;
        DimVector strides;
    };
    
    using meta_return_ty = precompute_out <true, true>;
    meta_return_ty meta(const Tensor & self, std::vector<otter::optional<otter::Tensor>> indices);
};

struct structured_prod_dim_int : public TensorIterator {
    void meta(const Tensor & self, int64_t dim, bool keepdim, ScalarType dtype);
};

struct structured_mean_dim : public TensorIterator {
    void meta(const Tensor & self, IntArrayRef dim, bool keepdim, ScalarType dtype);
};

struct structured_min_dim : public TensorIterator {
    
    template <bool DIM = false>
    struct precompute_out {
        precompute_out<true> set_dim(int64_t value) {
            static_assert(DIM == false, "dim already set");
            precompute_out<true> ret;
            ret.dim = value;
            return ret;
        }
                
        int64_t dim;
    };
    using meta_return_ty = precompute_out <true>;
    meta_return_ty meta(const Tensor & self, int64_t dim, bool keepdim);
};

struct structured_max_dim : public TensorIterator {
    
    template <bool DIM = false>
    struct precompute_out {
        precompute_out<true> set_dim(int64_t value) {
            static_assert(DIM == false, "dim already set");
            precompute_out<true> ret;
            ret.dim = value;
            return ret;
        }
                
        int64_t dim;
    };
    using meta_return_ty = precompute_out <true>;
    meta_return_ty meta(const Tensor & self, int64_t dim, bool keepdim);
};

struct structured_argmax : public TensorIterator {
    void meta(const Tensor & self, int64_t dim, bool keepdim);
};

struct structured_argmin : public TensorIterator {
    void meta(const Tensor & self, int64_t dim, bool keepdim);
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

struct structured_sigmoid_out : structured_sigmoid_Tensor {
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

struct structured_softmax_cpu_out : public structured_softmax {
    void impl(const Tensor & self, int64_t dim, bool half_to_float, const Tensor & out);
};

struct structured_sort_stable_out : public structured_sort_stable {
    void impl(const Tensor & self, bool stable, int64_t dim, bool descending, Tensor & values, Tensor & indices);
};

struct structured_topk_out_cpu : public structured_topk {
    void impl(const Tensor & self, int64_t k, int64_t dim, bool largest, bool sorted, Tensor & values,  Tensor & indices);
};

struct structured_scatter_src_out : public structured_scatter_src {
    void impl(const Tensor & self, int64_t dim, const Tensor & index, const Tensor & src, const Tensor & out);
};

struct structured_scatter_value_out : public structured_scatter_value {
    void impl(const Tensor & self, int64_t dim, const Tensor & index, const Scalar & value, const Tensor & out);
};

struct structured_scatter_reduce_out : public structured_scatter_reduce {
    void impl(const Tensor & self, int64_t dim, const Tensor & index, const Tensor & src, int64_t reduce, const Tensor & out);
};

struct structured_scatter_value_reduce_out : public structured_scatter_value_reduce {
    void impl(const Tensor & self, int64_t dim, const Tensor & index, const Scalar & value, int64_t reduce, const Tensor & out);
};

struct structured_gather_out : public structured_gather {
    void impl(const Tensor & self, int64_t dim, const Tensor & index, bool sparse_grad, const Tensor & out);
};

//struct structured_scatter_reduce_two_out : public structured_scatter_reduce_two {
//    void impl(const Tensor & self, int64_t dim, const Tensor & index, const Tensor & src, int64_t reduce, bool include_self, const Tensor & out);
//};
//
//struct structured_scatter_add_out : public structured_scatter_add {
//    void impl(const Tensor & self, int64_t dim, const Tensor & index, const Tensor & src, const Tensor & out);
//};

struct structured_bmm_out_cpu : public structured_bmm {
    void impl(const Tensor & self, const Tensor & mat2, const Tensor & out);
};

struct structured_baddbmm_out_cpu : public structured_baddbmm {
    void impl(const Tensor & self, const Tensor & batch1, const Tensor & batch2, const Scalar & beta, const Scalar & alpha, const Tensor & out);
};

struct structured_sum_out : public structured_sum_dim_IntList {
    void impl(const Tensor & self, IntArrayRef dim, bool keepdim, ScalarType dtype, const Tensor & out);
};

struct structured_index_out : public structured_index_Tensor {
    void impl(const Tensor & self, DimVector sizes, DimVector strides, const Tensor & out);
};

struct structured_prod_out : public structured_prod_dim_int {
    void impl(const Tensor & self, int64_t dim, bool keepdim, ScalarType dtype, const Tensor & out);
};

struct structured_mean_out : public structured_mean_dim {
    void impl(const Tensor & self, IntArrayRef dim, bool keepdim, ScalarType dtype, const Tensor & out);
};

struct structured_min_out : public structured_min_dim {
    void impl(const Tensor & self, int64_t dim, bool keepdim, const Tensor & min, const Tensor & min_indices);
};

struct structured_max_out : public structured_max_dim {
    void impl(const Tensor & self, int64_t dim, bool keepdim, const Tensor & max, const Tensor & max_values);
};

struct structured_argmin_out : public structured_argmin {
    void impl(const Tensor & self, int64_t dim, bool keepdim, const Tensor & out);
};

struct structured_argmax_out : public structured_argmax {
    void impl(const Tensor & self, int64_t dim, bool keepdim, const Tensor & out);
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

Tensor sigmoid(const Tensor & self);
Tensor & sigmoid_out(Tensor & out, const Tensor & self);
Tensor & sigmoid_(Tensor & self);

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

Tensor _softmax(const Tensor & self, int64_t dim, bool half_to_float);
Tensor & _softmax_out(Tensor & out, const Tensor & self, int64_t dim, bool half_to_float);
Tensor & _softmax_outf(const Tensor & self, int64_t dim, bool half_to_float, Tensor & out);

::std::tuple<Tensor, Tensor> sort(const Tensor & self, bool stable, int64_t dim, bool descending);
::std::tuple<Tensor &, Tensor &> sort_out(Tensor & values, Tensor & indices, const Tensor & self, bool stable, int64_t dim, bool descending);
::std::tuple<Tensor &,Tensor &> sort_outf(const Tensor & self, bool stable, int64_t dim, bool descending, Tensor & values, Tensor & indices);

::std::tuple<Tensor,Tensor> topk(const Tensor & self, int64_t k, int64_t dim, bool largest, bool sorted);
::std::tuple<Tensor &,Tensor &> topk_out(Tensor & values, Tensor & indices, const Tensor & self, int64_t k, int64_t dim, bool largest, bool sorted);
::std::tuple<Tensor &,Tensor &> topk_outf(const Tensor & self, int64_t k, int64_t dim, bool largest, bool sorted, Tensor & values, Tensor & indices);

Tensor scatter(const Tensor & self, int64_t dim, const Tensor & index, const Tensor & src);
Tensor & scatter_out(Tensor & out, const Tensor & self, int64_t dim, const Tensor & index, const Tensor & src);
Tensor & scatter_outf(const Tensor & self, int64_t dim, const Tensor & index, const Tensor & src, Tensor & out);
Tensor & scatter_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & src);
Tensor scatter(const Tensor & self, int64_t dim, const Tensor & index, const Scalar & value);
Tensor & scatter_out(Tensor & out, const Tensor & self, int64_t dim, const Tensor & index, const Scalar & value);
Tensor & scatter_outf(const Tensor & self, int64_t dim, const Tensor & index, const Scalar & value, Tensor & out);
Tensor & scatter_(Tensor & self, int64_t dim, const Tensor & index, const Scalar & value);
Tensor scatter(const Tensor & self, int64_t dim, const Tensor & index, const Tensor & src, int64_t reduce);
Tensor & scatter_out(Tensor & out, const Tensor & self, int64_t dim, const Tensor & index, const Tensor & src, int64_t reduce);
Tensor & scatter_outf(const Tensor & self, int64_t dim, const Tensor & index, const Tensor & src, int64_t reduce, Tensor & out);
Tensor & scatter_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & src, int64_t reduce);
Tensor scatter(const Tensor & self, int64_t dim, const Tensor & index, const Scalar & value, int64_t reduce);
Tensor & scatter_out(Tensor & out, const Tensor & self, int64_t dim, const Tensor & index, const Scalar & value, int64_t reduce);
Tensor & scatter_outf(const Tensor & self, int64_t dim, const Tensor & index, const Scalar & value, int64_t reduce, Tensor & out);
Tensor & scatter_(Tensor & self, int64_t dim, const Tensor & index, const Scalar & value, int64_t reduce);

Tensor baddbmm(const Tensor & self, const Tensor & batch1, const Tensor & batch2, const Scalar & beta, const Scalar & alpha);
Tensor & baddbmm_out(Tensor & out, const Tensor & self, const Tensor & batch1, const Tensor & batch2, const Scalar & beta, const Scalar & alpha);
Tensor & baddbmm_outf(const Tensor & self, const Tensor & batch1, const Tensor & batch2, const Scalar & beta, const Scalar & alpha, Tensor & out);
Tensor & baddbmm_(Tensor & self, const Tensor & batch1, const Tensor & batch2, const Scalar & beta, const Scalar & alpha);

Tensor bmm(const Tensor & self, const Tensor & mat2);
Tensor & bmm_out(Tensor & out, const Tensor & self, const Tensor & mat2);
Tensor & bmm_outf(const Tensor & self, const Tensor & mat2, Tensor & out);

Tensor sum(const Tensor & self, IntArrayRef dim, bool keepdim, ScalarType dtype);
Tensor & sum_out(Tensor & out, const Tensor & self, IntArrayRef dim, bool keepdim, ScalarType dtype);
Tensor & sum_outf(const Tensor & self, IntArrayRef dim, bool keepdim, ScalarType dtype, Tensor & out);

Tensor index(const Tensor & self, const std::vector<otter::optional<Tensor>> & indices);
Tensor & index_out(Tensor & out, const Tensor & self, const std::vector<otter::optional<Tensor>> & indices);
Tensor & index_outf(const Tensor & self, const std::vector<otter::optional<Tensor>> & indices, Tensor & out);

Tensor prod(const Tensor & self, int64_t dim, bool keepdim = false, ScalarType dtype = ScalarType::Undefined);
Tensor & prod_out(Tensor & out, const Tensor & self, int64_t dim, bool keepdim = false, ScalarType dtype = ScalarType::Undefined);
Tensor & prod_outf(const Tensor & self, int64_t dim, bool keepdim, ScalarType dtype, Tensor & out);

Tensor mean(const Tensor & self, IntArrayRef dim, bool keepdim = false, ScalarType dtype = ScalarType::Undefined);
Tensor & mean_out(Tensor & out, const Tensor & self, IntArrayRef dim, bool keepdim = false, ScalarType dtype = ScalarType::Undefined);
Tensor & mean_outf(const Tensor & self, IntArrayRef dim, bool keepdim, ScalarType dtype, Tensor & out);

}   // end namespace native

}   // end namespace otter

#endif /* TensorFunction_hpp */
