//
//  TensorFunction.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/1/31.
//

#include "TensorResize.hpp"
#include "EmptyTensor.hpp"
#include "TensorFunction.hpp"

namespace otter {

Tensor create_out(IntArrayRef sizes, IntArrayRef strides, TensorOptions options) {
    if (strides.empty()) {
        return otter::empty_cpu(sizes, options);
    } else {
        return otter::empty_strided_cpu(sizes, strides, options);
    }
}

void resize_out(const Tensor &out, IntArrayRef sizes, IntArrayRef strides, const TensorOptions &options) {
    (void)options;
    assert(out.dtype() == options.dtype());
    assert(out.device() == options.device());
    
    const bool resized = otter::native::resize_output(out, sizes);
    if (resized) {
        if (!strides.empty()) {
            
        }
    }
}

void check_inplace(const Tensor &self, IntArrayRef sizes, const TensorOptions &options) {
    (void)self;
    (void)sizes;
    (void)options;
    // "Bad in-place call: ", "input tensor dtype ", self.dtype(), " and output tensor dtype ", options.dtype(), " should match"
    assert(options.dtype() == self.dtype());
    // "Bad in-place call: ", "input tensor device ", self.device(), " and output tensor device ", options.device(), " should match"
    assert(options.device() == self.device());
    // "Bad in-place call: ", "input tensor size ", self.sizes(), " and output tensor size ", sizes, " should match"
    assert(sizes == self.sizes());
}

// add cpu
DEFINE_FINAL_OP_AFTER(add_out)
Tensor wrapper_add_Tensor(const Tensor & self, const Tensor & other, const Scalar & alpha) {
    structured_add_out_functional op;
    op.meta(self, other, alpha);
    op.impl(self, other, alpha, *op.outputs_[0]);
    return std::move(op.outputs_[0]).take();
}

Tensor & wrapper_add_out_out(const Tensor & self, const Tensor & other, const Scalar & alpha, Tensor & out) {
    structured_add_out_out op(out);
    op.meta(self, other, alpha);
    op.impl(self, other, alpha, op.outputs_[0]);
    return out;
}

Tensor & wrapper_add__Tensor(Tensor & self, const Tensor & other, const Scalar & alpha) {
    structured_add_out_inplace op(self);
    op.meta(self, other, alpha);
    op.impl(self, other, alpha, op.outputs_[0]);
    return self;
}
// end add cpu

// sub cpu
DEFINE_FINAL_OP_AFTER(sub_out)
Tensor wrapper_sub_Tensor(const Tensor & self, const Tensor & other, const Scalar & alpha) {
    structured_sub_out_functional op;
    op.meta(self, other, alpha);
    op.impl(self, other, alpha, *op.outputs_[0]);
    return std::move(op.outputs_[0]).take();
}

Tensor & wrapper_sub_out_out(const Tensor & self, const Tensor & other, const Scalar & alpha, Tensor & out) {
    structured_sub_out_out op(out);
    op.meta(self, other, alpha);
    op.impl(self, other, alpha, op.outputs_[0]);
    return out;
}

Tensor & wrapper_sub__Tensor(Tensor & self, const Tensor & other, const Scalar & alpha) {
    structured_sub_out_inplace op(self);
    op.meta(self, other, alpha);
    op.impl(self, other, alpha, op.outputs_[0]);
    return self;
}
// end sub cpu

// mul cpu
DEFINE_FINAL_OP_AFTER(mul_out)
Tensor wrapper_mul_Tensor(const Tensor & self, const Tensor & other) {
    structured_mul_out_functional op;
    op.meta(self, other);
    op.impl(self, other, *op.outputs_[0]);
    return std::move(op.outputs_[0]).take();
}

Tensor & wrapper_mul_out_out(const Tensor & self, const Tensor & other, Tensor & out) {
    structured_mul_out_out op(out);
    op.meta(self, other);
    op.impl(self, other, op.outputs_[0]);
    return out;
}

Tensor & wrapper_mul__Tensor(Tensor & self, const Tensor & other) {
    structured_mul_out_inplace op(self);
    op.meta(self, other);
    op.impl(self, other, op.outputs_[0]);
    return self;
}
// end mul cpu

// div cpu
DEFINE_FINAL_OP_AFTER(div_out)
Tensor wrapper_div_Tensor(const Tensor & self, const Tensor & other) {
    structured_div_out_functional op;
    op.meta(self, other);
    op.impl(self, other, *op.outputs_[0]);
    return std::move(op.outputs_[0]).take();
}

Tensor & wrapper_div_out_out(const Tensor & self, const Tensor & other, Tensor & out) {
    structured_div_out_out op(out);
    op.meta(self, other);
    op.impl(self, other, op.outputs_[0]);
    return out;
}

Tensor & wrapper_div__Tensor(Tensor & self, const Tensor & other) {
    structured_div_out_inplace op(self);
    op.meta(self, other);
    op.impl(self, other, op.outputs_[0]);
    return self;
}
// end div cpu

// remainder cpu
DEFINE_FINAL_OP_AFTER(remainder_out)
Tensor wrapper_remainder_Tensor(const Tensor & self, const Tensor & other) {
    structured_remainder_out_functional op;
    op.meta(self, other);
    op.impl(self, other, *op.outputs_[0]);
    return std::move(op.outputs_[0]).take();
}

Tensor & wrapper_remainder_out_out(const Tensor & self, const Tensor & other, Tensor & out) {
    structured_remainder_out_out op(out);
    op.meta(self, other);
    op.impl(self, other, op.outputs_[0]);
    return out;
}

Tensor & wrapper_remainder__Tensor(Tensor & self, const Tensor & other) {
    structured_remainder_out_inplace op(self);
    op.meta(self, other);
    op.impl(self, other, op.outputs_[0]);
    return self;
}
// end remainder cpu

// bitwise_and cpu
DEFINE_FINAL_OP_AFTER(bitwise_and_out)
Tensor wrapper_bitwise_and_Tensor(const Tensor & self, const Tensor & other) {
    structured_bitwise_and_out_functional op;
    op.meta(self, other);
    op.impl(self, other, *op.outputs_[0]);
    return std::move(op.outputs_[0]).take();
}

Tensor & wrapper_bitwise_and_out_out(const Tensor & self, const Tensor & other, Tensor & out) {
    structured_bitwise_and_out_out op(out);
    op.meta(self, other);
    op.impl(self, other, op.outputs_[0]);
    return out;
}

Tensor & wrapper_bitwise_and__Tensor(Tensor & self, const Tensor & other) {
    structured_bitwise_and_out_inplace op(self);
    op.meta(self, other);
    op.impl(self, other, op.outputs_[0]);
    return self;
}
// end bitwise_and cpu

// bitwise_or cpu
DEFINE_FINAL_OP_AFTER(bitwise_or_out)
Tensor wrapper_bitwise_or_Tensor(const Tensor & self, const Tensor & other) {
    structured_bitwise_or_out_functional op;
    op.meta(self, other);
    op.impl(self, other, *op.outputs_[0]);
    return std::move(op.outputs_[0]).take();
}

Tensor & wrapper_bitwise_or_out_out(const Tensor & self, const Tensor & other, Tensor & out) {
    structured_bitwise_or_out_out op(out);
    op.meta(self, other);
    op.impl(self, other, op.outputs_[0]);
    return out;
}

Tensor & wrapper_bitwise_or__Tensor(Tensor & self, const Tensor & other) {
    structured_bitwise_or_out_inplace op(self);
    op.meta(self, other);
    op.impl(self, other, op.outputs_[0]);
    return self;
}
// end bitwise_or cpu

// bitwise_xor cpu
DEFINE_FINAL_OP_AFTER(bitwise_xor_out)
Tensor wrapper_bitwise_xor_Tensor(const Tensor & self, const Tensor & other) {
    structured_bitwise_xor_out_functional op;
    op.meta(self, other);
    op.impl(self, other, *op.outputs_[0]);
    return std::move(op.outputs_[0]).take();
}

Tensor & wrapper_bitwise_xor_out_out(const Tensor & self, const Tensor & other, Tensor & out) {
    structured_bitwise_xor_out_out op(out);
    op.meta(self, other);
    op.impl(self, other, op.outputs_[0]);
    return out;
}

Tensor & wrapper_bitwise_xor__Tensor(Tensor & self, const Tensor & other) {
    structured_bitwise_xor_out_inplace op(self);
    op.meta(self, other);
    op.impl(self, other, op.outputs_[0]);
    return self;
}
// end bitwise_and cpu

// neg cpu
DEFINE_FINAL_OP_AFTER(neg_out)
Tensor wrapper_neg(const Tensor & self) {
    structured_neg_out_functional op;
    op.meta(self);
    op.impl(self, *op.outputs_[0]);
    return std::move(op.outputs_[0]).take();
}

Tensor & wrapper_neg_out(const Tensor & self, Tensor & out) {
    structured_neg_out_out op(out);
    op.meta(self);
    op.impl(self, op.outputs_[0]);
    return out;
}

Tensor & wrapper_neg_(Tensor & self) {
    structured_neg_out_inplace op(self);
    op.meta(self);
    op.impl(self, op.outputs_[0]);
    return self;
}

// end neg cpu

// bitwise_not cpu
DEFINE_FINAL_OP_AFTER(bitwise_not_out)
Tensor wrapper_bitwise_not(const Tensor & self) {
    structured_bitwise_not_out_functional op;
    op.meta(self);
    op.impl(self, *op.outputs_[0]);
    return std::move(op.outputs_[0]).take();
}

Tensor & wrapper_bitwise_not_out(const Tensor & self, Tensor & out) {
    structured_bitwise_not_out_out op(out);
    op.meta(self);
    op.impl(self, op.outputs_[0]);
    return out;
}

Tensor & wrapper_bitwise_not_(Tensor & self) {
    structured_bitwise_not_out_inplace op(self);
    op.meta(self);
    op.impl(self, op.outputs_[0]);
    return self;
}

// end bitwise_not cpu

// abs cpu
DEFINE_FINAL_OP_AFTER(abs_out)
Tensor wrapper_abs(const Tensor & self) {
    structured_abs_out_functional op;
    op.meta(self);
    op.impl(self, *op.outputs_[0]);
    return std::move(op.outputs_[0]).take();
}

Tensor & wrapper_abs_out(const Tensor & self, Tensor & out) {
    structured_abs_out_out op(out);
    op.meta(self);
    op.impl(self, op.outputs_[0]);
    return out;
}

Tensor & wrapper_abs_(Tensor & self) {
    structured_abs_out_inplace op(self);
    op.meta(self);
    op.impl(self, op.outputs_[0]);
    return self;
}

// end abs cpu

// sin cpu
DEFINE_FINAL_OP_AFTER(sin_out)
Tensor wrapper_sin(const Tensor & self) {
    structured_sin_out_functional op;
    op.meta(self);
    op.impl(self, *op.outputs_[0]);
    return std::move(op.outputs_[0]).take();
}

Tensor & wrapper_sin_out(const Tensor & self, Tensor & out) {
    structured_sin_out_out op(out);
    op.meta(self);
    op.impl(self, op.outputs_[0]);
    return out;
}

Tensor & wrapper_sin_(Tensor & self) {
    structured_sin_out_inplace op(self);
    op.meta(self);
    op.impl(self, op.outputs_[0]);
    return self;
}

// end sin cpu

// cos cpu
DEFINE_FINAL_OP_AFTER(cos_out)
Tensor wrapper_cos(const Tensor & self) {
    structured_cos_out_functional op;
    op.meta(self);
    op.impl(self, *op.outputs_[0]);
    return std::move(op.outputs_[0]).take();
}

Tensor & wrapper_cos_out(const Tensor & self, Tensor & out) {
    structured_cos_out_out op(out);
    op.meta(self);
    op.impl(self, op.outputs_[0]);
    return out;
}

Tensor & wrapper_cos_(Tensor & self) {
    structured_cos_out_inplace op(self);
    op.meta(self);
    op.impl(self, op.outputs_[0]);
    return self;
}

// end cos cpu

// tan cpu
DEFINE_FINAL_OP_AFTER(tan_out)
Tensor wrapper_tan(const Tensor & self) {
    structured_tan_out_functional op;
    op.meta(self);
    op.impl(self, *op.outputs_[0]);
    return std::move(op.outputs_[0]).take();
}

Tensor & wrapper_tan_out(const Tensor & self, Tensor & out) {
    structured_tan_out_out op(out);
    op.meta(self);
    op.impl(self, op.outputs_[0]);
    return out;
}

Tensor & wrapper_tan_(Tensor & self) {
    structured_tan_out_inplace op(self);
    op.meta(self);
    op.impl(self, op.outputs_[0]);
    return self;
}

// end tan cpu

// exp cpu
DEFINE_FINAL_OP_AFTER(exp_out)
Tensor wrapper_exp(const Tensor & self) {
    structured_exp_out_functional op;
    op.meta(self);
    op.impl(self, *op.outputs_[0]);
    return std::move(op.outputs_[0]).take();
}

Tensor & wrapper_exp_out(const Tensor & self, Tensor & out) {
    structured_exp_out_out op(out);
    op.meta(self);
    op.impl(self, op.outputs_[0]);
    return out;
}

Tensor & wrapper_exp_(Tensor & self) {
    structured_exp_out_inplace op(self);
    op.meta(self);
    op.impl(self, op.outputs_[0]);
    return self;
}

// end exp cpu

// sqrt cpu
DEFINE_FINAL_OP_AFTER(sqrt_out)
Tensor wrapper_sqrt(const Tensor & self) {
    structured_sqrt_out_functional op;
    op.meta(self);
    op.impl(self, *op.outputs_[0]);
    return std::move(op.outputs_[0]).take();
}

Tensor & wrapper_sqrt_out(const Tensor & self, Tensor & out) {
    structured_sqrt_out_out op(out);
    op.meta(self);
    op.impl(self, op.outputs_[0]);
    return out;
}

Tensor & wrapper_sqrt_(Tensor & self) {
    structured_sqrt_out_inplace op(self);
    op.meta(self);
    op.impl(self, op.outputs_[0]);
    return self;
}

// end sqrt cpu

// sigmoid cpu
DEFINE_FINAL_OP_AFTER(sigmoid_out)
Tensor wrapper_sigmoid(const Tensor & self) {
    structured_sigmoid_out_functional op;
    op.meta(self);
    op.impl(self, *op.outputs_[0]);
    return std::move(op.outputs_[0]).take();
}

Tensor & wrapper_sigmoid_out(const Tensor & self, Tensor & out) {
    structured_sigmoid_out_out op(out);
    op.meta(self);
    op.impl(self, op.outputs_[0]);
    return out;
}

Tensor & wrapper_sigmoid_(Tensor & self) {
    structured_sigmoid_out_inplace op(self);
    op.meta(self);
    op.impl(self, op.outputs_[0]);
    return self;
}

// end sigmoid cpu

// addmm cpu
struct structured_addmm_out_cpu_functional : structured_addmm_out_cpu {
    void set_output(int64_t output_idx, IntArrayRef sizes, IntArrayRef strides, TensorOptions options) override {
        outputs_[output_idx] = create_out(sizes, strides, options);
    }
    const Tensor& maybe_get_output(int64_t output_idx) override {
        return *outputs_[output_idx];
    }
    std::array<otter::ExclusivelyOwned<Tensor>, 1> outputs_;
};

Tensor wrapper_addmm(const Tensor& self, const Tensor& mat1, const Tensor& mat2, const Scalar& beta, const Scalar& alpha) {
    structured_addmm_out_cpu_functional op;
    op.meta(self, mat1, mat2, beta, alpha);
    op.impl(self, mat1, mat2, beta, alpha, *op.outputs_[0]);
    
    return std::move(op.outputs_[0]).take();
}

struct structured_addmm_out_cpu_out : structured_addmm_out_cpu {
    structured_addmm_out_cpu_out(Tensor& out0) : outputs_{ std::ref(out0) } {}
    void set_output(int64_t output_idx, IntArrayRef sizes, IntArrayRef strides, TensorOptions options) override {
        const auto& out = outputs_[output_idx].get();
        otter::resize_out(out, sizes, strides, options);
    }
    const Tensor& maybe_get_output(int64_t output_idx) override {
        return outputs_[output_idx];
    }
    std::array<std::reference_wrapper<Tensor>, 1> outputs_;
};

Tensor & wrapper_addmm_out(const Tensor & self, const Tensor & mat1, const Tensor & mat2, const Scalar & beta, const Scalar & alpha, Tensor & out) {
    structured_addmm_out_cpu_out op(out);
    op.meta(self, mat1, mat2, beta, alpha);
    op.impl(self, mat1, mat2, beta, alpha, op.outputs_[0]);
    return out;
}

struct structured_addmm_out_cpu_inplace : structured_addmm_out_cpu {
    structured_addmm_out_cpu_inplace(Tensor& out0) : outputs_{ std::ref(out0) } {}
    void set_output(int64_t output_idx, IntArrayRef sizes, IntArrayRef /*strides*/, TensorOptions options) override {
        const auto& out = outputs_[output_idx].get();
        check_inplace(out, sizes, options);
    }
    const Tensor& maybe_get_output(int64_t output_idx) override {
        return outputs_[output_idx];
    }
    std::array<std::reference_wrapper<Tensor>, 1> outputs_;
};

Tensor & wrapper_addmm_(Tensor & self, const Tensor & mat1, const Tensor & mat2, const Scalar & beta, const Scalar & alpha) {
    structured_addmm_out_cpu_inplace op(self);
    op.meta(self, mat1, mat2, beta, alpha);
    op.impl(self, mat1, mat2, beta, alpha, op.outputs_[0]);
    return self;
}

// end addmm cpu

// addmm cpu
struct structured_mm_out_cpu_functional : structured_mm_out_cpu {
    void set_output(int64_t output_idx, IntArrayRef sizes, IntArrayRef strides, TensorOptions options) override {
        outputs_[output_idx] = create_out(sizes, strides, options);
    }
    const Tensor& maybe_get_output(int64_t output_idx) override {
        return *outputs_[output_idx];
    }
    std::array<otter::ExclusivelyOwned<Tensor>, 1> outputs_;
};

Tensor wrapper_mm(const Tensor& self, const Tensor& other) {
    structured_mm_out_cpu_functional op;
    op.meta(self, other);
    op.impl(self, other, *op.outputs_[0]);
    
    return std::move(op.outputs_[0]).take();
}

struct structured_mm_out_cpu_out : structured_mm_out_cpu {
    structured_mm_out_cpu_out(Tensor& out0) : outputs_{ std::ref(out0) } {}
    void set_output(int64_t output_idx, IntArrayRef sizes, IntArrayRef strides, TensorOptions options) override {
        const auto& out = outputs_[output_idx].get();
        otter::resize_out(out, sizes, strides, options);
    }
    const Tensor& maybe_get_output(int64_t output_idx) override {
        return outputs_[output_idx];
    }
    std::array<std::reference_wrapper<Tensor>, 1> outputs_;
};

Tensor & wrapper_mm_out(const Tensor & self, const Tensor & other, Tensor & out) {
    structured_mm_out_cpu_out op(out);
    op.meta(self, other);
    op.impl(self, other, op.outputs_[0]);
    return out;
}

struct structured_mm_out_cpu_inplace : structured_mm_out_cpu {
    structured_mm_out_cpu_inplace(Tensor& out0) : outputs_{ std::ref(out0) } {}
    void set_output(int64_t output_idx, IntArrayRef sizes, IntArrayRef /*strides*/, TensorOptions options) override {
        const auto& out = outputs_[output_idx].get();
        check_inplace(out, sizes, options);
    }
    const Tensor& maybe_get_output(int64_t output_idx) override {
        return outputs_[output_idx];
    }
    std::array<std::reference_wrapper<Tensor>, 1> outputs_;
};

Tensor & wrapper_mm_(Tensor & self, const Tensor & other) {
    structured_mm_out_cpu_inplace op(self);
    op.meta(self, other);
    op.impl(self, other, op.outputs_[0]);
    return self;
}
// end addmm cpu

// max_pool2d
struct structured_max_pool2d_with_indices_out_cpu_functional final : public structured_max_pool2d_with_indices_out_cpu {
    
    void set_output(int64_t output_idx, IntArrayRef sizes, IntArrayRef strides, TensorOptions options) override {
        outputs_[output_idx] = create_out(sizes, strides, options);
    }
    const Tensor& maybe_get_output(int64_t output_idx) override {
        return *outputs_[output_idx];
    }
    std::array<ExclusivelyOwned<Tensor>, 2> outputs_;
};

std::tuple<Tensor, Tensor> wrapper_max_pool2d_with_indices(const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode) {
    structured_max_pool2d_with_indices_out_cpu_functional op;
    op.meta(self, kernel_size, stride, padding, dilation, ceil_mode);
    op.impl(self, kernel_size, stride, padding, dilation, ceil_mode, *op.outputs_[0], *op.outputs_[1]);
    return std::make_tuple(std::move(op.outputs_[0]).take(), std::move(op.outputs_[1]).take());
}

struct structured_max_pool2d_with_indices_out_cpu_out final : public structured_max_pool2d_with_indices_out_cpu {
    structured_max_pool2d_with_indices_out_cpu_out(Tensor& out0, Tensor& out1) : outputs_{ std::ref(out0), std::ref(out1) } {}
    
    void set_output(int64_t output_idx, IntArrayRef sizes, IntArrayRef strides, TensorOptions options) override {
        const auto& out = outputs_[output_idx].get();
        resize_out(out, sizes, strides, options);
    }
    
    const Tensor& maybe_get_output(int64_t output_idx) override {
        return outputs_[output_idx];
    }
    std::array<std::reference_wrapper<Tensor>, 2> outputs_;
};

std::tuple<Tensor &, Tensor &> wrapper_max_pool2d_with_indices_out_out(const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode, Tensor & out, Tensor & indices) {
    structured_max_pool2d_with_indices_out_cpu_out op(out, indices);
    op.meta(self, kernel_size, stride, padding, dilation, ceil_mode);
    op.impl(self, kernel_size, stride, padding, dilation, ceil_mode, op.outputs_[0], op.outputs_[1]);
    return std::forward_as_tuple(out, indices);
}

// end max_pool2d

// leaky_relu cpu
DEFINE_FINAL_OP_AFTER(leaky_relu_out)
Tensor wrapper_leaky_relu(const Tensor & self, const Scalar & negative_slope) {
    structured_leaky_relu_out_functional op;
    op.meta(self, negative_slope);
    op.impl(self, negative_slope, *op.outputs_[0]);
    return std::move(op.outputs_[0]).take();
}

Tensor & wrapper_leaky_relu_out(const Tensor & self, const Scalar & negative_slope, Tensor & out) {
    structured_leaky_relu_out_out op(out);
    op.meta(self, negative_slope);
    op.impl(self, negative_slope, op.outputs_[0]);
    return out;
}

Tensor & wrapper_leaky_relu_(Tensor & self, const Scalar & negative_slope) {
    structured_leaky_relu_out_inplace op(self);
    op.meta(self, negative_slope);
    op.impl(self, negative_slope, op.outputs_[0]);
    return self;
}
// end leaky_relu cpu

// upsample nearest
struct structured_upsample_nearest2d_out_cpu_functional final : public structured_upsample_nearest2d_out_cpu {
    
    void set_output(int64_t output_idx, IntArrayRef sizes, IntArrayRef strides, TensorOptions options) override {
        outputs_[output_idx] = create_out(sizes, strides, options);
    }
    
    const Tensor& maybe_get_output(int64_t output_idx) override {
        return *outputs_[output_idx];
    }
    std::array<ExclusivelyOwned<Tensor>, 1> outputs_;
};

Tensor wrapper_upsample_nearest2d(const Tensor & self, IntArrayRef output_size, double scales_h, double scales_w) {
    structured_upsample_nearest2d_out_cpu_functional op;
    op.meta(self, output_size, scales_h, scales_w);
    op.impl(self, output_size, scales_h, scales_w, *op.outputs_[0]);
    return std::move(op.outputs_[0]).take();
}

struct structured_upsample_nearest2d_out_cpu_out final : public structured_upsample_nearest2d_out_cpu {
    structured_upsample_nearest2d_out_cpu_out(Tensor& out0) : outputs_{ std::ref(out0) } {}
    
    void set_output(int64_t output_idx, IntArrayRef sizes, IntArrayRef strides, TensorOptions options) override {
        const auto& out = outputs_[output_idx].get();
        resize_out(out, sizes, strides, options);
    }
    
    const Tensor& maybe_get_output(int64_t output_idx) override {
        return outputs_[output_idx];
    }
    std::array<std::reference_wrapper<Tensor>, 1> outputs_;
};

Tensor & wrapper_upsample_nearest2d_out_out(const Tensor & self, IntArrayRef output_size, double scales_h, double scales_w, Tensor & out) {
    structured_upsample_nearest2d_out_cpu_out op(out);
    op.meta(self, output_size, scales_h, scales_w);
    op.impl(self, output_size, scales_h, scales_w, op.outputs_[0]);
    return out;
}

// end upsample nearest

// upsample bilinear
struct structured_upsample_bilinear2d_out_cpu_functional final : public structured_upsample_bilinear2d_out_cpu {
    
    void set_output(int64_t output_idx, IntArrayRef sizes, IntArrayRef strides, TensorOptions options) override {
        
        outputs_[output_idx] = create_out(sizes, strides, options);
    }
    
    const Tensor& maybe_get_output(int64_t output_idx) override {
        return *outputs_[output_idx];
    }
    std::array<ExclusivelyOwned<Tensor>, 1> outputs_;
};

Tensor wrapper_upsample_bilinear2d(const Tensor & self, IntArrayRef output_size, bool align_corners, double scales_h, double scales_w) {
    structured_upsample_bilinear2d_out_cpu_functional op;
    op.meta(self, output_size, align_corners, scales_h, scales_w);
    op.impl(self, output_size, align_corners, scales_h, scales_w, *op.outputs_[0]);
    
    return std::move(op.outputs_[0]).take();
}

struct structured_upsample_bilinear2d_out_cpu_out final : public structured_upsample_bilinear2d_out_cpu {
    structured_upsample_bilinear2d_out_cpu_out(Tensor& out0) : outputs_{ std::ref(out0) } {}
    
    void set_output(int64_t output_idx, IntArrayRef sizes, IntArrayRef strides, TensorOptions options) override {
        
        const auto& out = outputs_[output_idx].get();
        resize_out(out, sizes, strides, options);
    }
    
    const Tensor& maybe_get_output(int64_t output_idx) override {
        return outputs_[output_idx];
    }
    std::array<std::reference_wrapper<Tensor>, 1> outputs_;
};

Tensor & wrapper_upsample_bilinear2d_out_out(const Tensor & self, IntArrayRef output_size, bool align_corners, double scales_h, double scales_w, Tensor & out) {
    structured_upsample_bilinear2d_out_cpu_out op(out);
    op.meta(self, output_size, align_corners, scales_h, scales_w);
    op.impl(self, output_size, align_corners, scales_h, scales_w, op.outputs_[0]);
    
    return out;
}
// end upsample bilinear

// threshold
struct structured_threshold_out_functional final : public structured_threshold_out {
    
    void set_output(int64_t output_idx, IntArrayRef sizes, IntArrayRef strides, TensorOptions options) override {
        
        outputs_[output_idx] = create_out(sizes, strides, options);
        
        structured_threshold_out::set_output(output_idx, sizes, strides, options);
    }
    
    const Tensor& maybe_get_output(int64_t output_idx) override {
        return *outputs_[output_idx];
    }
    std::array<otter::ExclusivelyOwned<Tensor>, 1> outputs_;
};

otter::Tensor wrapper_threshold(const otter::Tensor & self, const otter::Scalar & threshold, const otter::Scalar & value) {
    structured_threshold_out_functional op;
    op.meta(self, threshold, value);
    op.impl(self, threshold, value, *op.outputs_[0]);
    return std::move(op.outputs_[0]).take();
}
struct structured_threshold_out_out final : public structured_threshold_out {
    structured_threshold_out_out(Tensor& out0) : outputs_{ std::ref(out0) } {}
    
    void set_output(int64_t output_idx, IntArrayRef sizes, IntArrayRef strides, TensorOptions options) override {
        
        const auto& out = outputs_[output_idx].get();
        resize_out(out, sizes, strides, options);
        
        structured_threshold_out::set_output(output_idx, sizes, strides, options);
    }
    
    const Tensor& maybe_get_output(int64_t output_idx) override {
        return outputs_[output_idx];
    }
    std::array<std::reference_wrapper<Tensor>, 1> outputs_;
};

otter::Tensor & wrapper_threshold_out_out(const otter::Tensor & self, const otter::Scalar & threshold, const otter::Scalar & value, otter::Tensor & out) {
    structured_threshold_out_out op(out);
    op.meta(self, threshold, value);
    op.impl(self, threshold, value, op.outputs_[0]);
    return out;
}
struct structured_threshold_out_inplace final : public structured_threshold_out {
    structured_threshold_out_inplace(Tensor& self) : outputs_{std::ref(self)} {}
    
    void set_output(int64_t output_idx, IntArrayRef sizes, IntArrayRef strides, TensorOptions options) override {
        
        const auto& out = outputs_[output_idx].get();
        check_inplace(out, sizes, options);
        
        structured_threshold_out::set_output(output_idx, sizes, strides, options);
    }
    
    const Tensor& maybe_get_output(int64_t output_idx) override {
        return outputs_[output_idx];
    }
    std::array<std::reference_wrapper<Tensor>, 1> outputs_;
};

otter::Tensor & wrapper_threshold_(otter::Tensor & self, const otter::Scalar & threshold, const otter::Scalar & value) {
    structured_threshold_out_inplace op(self);
    op.meta(self, threshold, value);
    op.impl(self, threshold, value, op.outputs_[0]);
    return self;
}
// end threshold

struct structured_eq_Scalar_out_functional final : public structured_eq_Scalar_out {
    void set_output(int64_t output_idx, IntArrayRef sizes, IntArrayRef strides,
                    TensorOptions options) override {
        outputs_[output_idx] = create_out(sizes, strides, options);
        
        structured_eq_Scalar_out::set_output(output_idx, sizes, strides, options);
    }
    const Tensor& maybe_get_output(int64_t output_idx) override {
        return *outputs_[output_idx];
    }
    std::array<ExclusivelyOwned<Tensor>, 1> outputs_;
};
Tensor wrapper_eq_Scalar(const Tensor & self, const Scalar & other) {
    structured_eq_Scalar_out_functional op;
    op.meta(self, other);
    op.impl(self, other, *op.outputs_[0]);
    return std::move(op.outputs_[0]).take();
}
struct structured_eq_Scalar_out_out final : public structured_eq_Scalar_out {
    structured_eq_Scalar_out_out(Tensor& out0) : outputs_{ std::ref(out0) } {}
    void set_output(int64_t output_idx, IntArrayRef sizes, IntArrayRef strides,
                    TensorOptions options) override {
        const auto& out = outputs_[output_idx].get();
        resize_out(out, sizes, strides, options);
        
        structured_eq_Scalar_out::set_output(output_idx, sizes, strides, options);
    }
    const Tensor& maybe_get_output(int64_t output_idx) override {
        return outputs_[output_idx];
    }
    std::array<std::reference_wrapper<Tensor>, 1> outputs_;
};
Tensor & wrapper_eq_out_Scalar_out(const Tensor & self, const Scalar & other, Tensor & out) {
    structured_eq_Scalar_out_out op(out);
    op.meta(self, other);
    op.impl(self, other, op.outputs_[0]);
    return out;
}
struct structured_eq_Scalar_out_inplace final : public structured_eq_Scalar_out {
    structured_eq_Scalar_out_inplace(Tensor& self) : outputs_{std::ref(self)} {}
    void set_output(int64_t output_idx, IntArrayRef sizes, IntArrayRef strides,
                    TensorOptions options) override {
        const auto& out = outputs_[output_idx].get();
        check_inplace(out, sizes, options);
        
        structured_eq_Scalar_out::set_output(output_idx, sizes, strides, options);
    }
    const Tensor& maybe_get_output(int64_t output_idx) override {
        return outputs_[output_idx];
    }
    std::array<std::reference_wrapper<Tensor>, 1> outputs_;
};
Tensor & wrapper_eq__Scalar(Tensor & self, const Scalar & other) {
    structured_eq_Scalar_out_inplace op(self);
    op.meta(self, other);
    op.impl(self, other, op.outputs_[0]);
    return self;
}
struct structured_eq_Tensor_out_functional final : public structured_eq_Tensor_out {
    void set_output(int64_t output_idx, IntArrayRef sizes, IntArrayRef strides,
                    TensorOptions options) override {
        outputs_[output_idx] = create_out(sizes, strides, options);
        
        structured_eq_Tensor_out::set_output(output_idx, sizes, strides, options);
    }
    const Tensor& maybe_get_output(int64_t output_idx) override {
        return *outputs_[output_idx];
    }
    std::array<ExclusivelyOwned<Tensor>, 1> outputs_;
};
Tensor wrapper_eq_Tensor(const Tensor & self, const Tensor & other) {
    structured_eq_Tensor_out_functional op;
    op.meta(self, other);
    op.impl(self, other, *op.outputs_[0]);
    return std::move(op.outputs_[0]).take();
}
struct structured_eq_Tensor_out_out final : public structured_eq_Tensor_out {
    structured_eq_Tensor_out_out(Tensor& out0) : outputs_{ std::ref(out0) } {}
    void set_output(int64_t output_idx, IntArrayRef sizes, IntArrayRef strides,
                    TensorOptions options) override {
        const auto& out = outputs_[output_idx].get();
        resize_out(out, sizes, strides, options);
        
        structured_eq_Tensor_out::set_output(output_idx, sizes, strides, options);
    }
    const Tensor& maybe_get_output(int64_t output_idx) override {
        return outputs_[output_idx];
    }
    std::array<std::reference_wrapper<Tensor>, 1> outputs_;
};
Tensor & wrapper_eq_out_Tensor_out(const Tensor & self, const Tensor & other, Tensor & out) {
    structured_eq_Tensor_out_out op(out);
    op.meta(self, other);
    op.impl(self, other, op.outputs_[0]);
    return out;
}
struct structured_eq_Tensor_out_inplace final : public structured_eq_Tensor_out {
    structured_eq_Tensor_out_inplace(Tensor& self) : outputs_{std::ref(self)} {}
    void set_output(int64_t output_idx, IntArrayRef sizes, IntArrayRef strides,
                    TensorOptions options) override {
        const auto& out = outputs_[output_idx].get();
        check_inplace(out, sizes, options);
        
        structured_eq_Tensor_out::set_output(output_idx, sizes, strides, options);
    }
    const Tensor& maybe_get_output(int64_t output_idx) override {
        return outputs_[output_idx];
    }
    std::array<std::reference_wrapper<Tensor>, 1> outputs_;
};
Tensor & wrapper_eq__Tensor(Tensor & self, const Tensor & other) {
    structured_eq_Tensor_out_inplace op(self);
    op.meta(self, other);
    op.impl(self, other, op.outputs_[0]);
    return self;
}

struct structured_ne_Scalar_out_functional final : public structured_ne_Scalar_out {
    void set_output(int64_t output_idx, IntArrayRef sizes, IntArrayRef strides,
                    TensorOptions options) override {
        outputs_[output_idx] = create_out(sizes, strides, options);
        
        structured_ne_Scalar_out::set_output(output_idx, sizes, strides, options);
    }
    const Tensor& maybe_get_output(int64_t output_idx) override {
        return *outputs_[output_idx];
    }
    std::array<ExclusivelyOwned<Tensor>, 1> outputs_;
};
Tensor wrapper_ne_Scalar(const Tensor & self, const Scalar & other) {
    structured_ne_Scalar_out_functional op;
    op.meta(self, other);
    op.impl(self, other, *op.outputs_[0]);
    return std::move(op.outputs_[0]).take();
}
struct structured_ne_Scalar_out_out final : public structured_ne_Scalar_out {
    structured_ne_Scalar_out_out(Tensor& out0) : outputs_{ std::ref(out0) } {}
    void set_output(int64_t output_idx, IntArrayRef sizes, IntArrayRef strides,
                    TensorOptions options) override {
        const auto& out = outputs_[output_idx].get();
        resize_out(out, sizes, strides, options);
        
        structured_ne_Scalar_out::set_output(output_idx, sizes, strides, options);
    }
    const Tensor& maybe_get_output(int64_t output_idx) override {
        return outputs_[output_idx];
    }
    std::array<std::reference_wrapper<Tensor>, 1> outputs_;
};
Tensor & wrapper_ne_out_Scalar_out(const Tensor & self, const Scalar & other, Tensor & out) {
    structured_ne_Scalar_out_out op(out);
    op.meta(self, other);
    op.impl(self, other, op.outputs_[0]);
    return out;
}
struct structured_ne_Scalar_out_inplace final : public structured_ne_Scalar_out {
    structured_ne_Scalar_out_inplace(Tensor& self) : outputs_{std::ref(self)} {}
    void set_output(int64_t output_idx, IntArrayRef sizes, IntArrayRef strides,
                    TensorOptions options) override {
        const auto& out = outputs_[output_idx].get();
        check_inplace(out, sizes, options);
        
        structured_ne_Scalar_out::set_output(output_idx, sizes, strides, options);
    }
    const Tensor& maybe_get_output(int64_t output_idx) override {
        return outputs_[output_idx];
    }
    std::array<std::reference_wrapper<Tensor>, 1> outputs_;
};
Tensor & wrapper_ne__Scalar(Tensor & self, const Scalar & other) {
    structured_ne_Scalar_out_inplace op(self);
    op.meta(self, other);
    op.impl(self, other, op.outputs_[0]);
    return self;
}
struct structured_ne_Tensor_out_functional final : public structured_ne_Tensor_out {
    void set_output(int64_t output_idx, IntArrayRef sizes, IntArrayRef strides,
                    TensorOptions options) override {
        outputs_[output_idx] = create_out(sizes, strides, options);
        
        structured_ne_Tensor_out::set_output(output_idx, sizes, strides, options);
    }
    const Tensor& maybe_get_output(int64_t output_idx) override {
        return *outputs_[output_idx];
    }
    std::array<ExclusivelyOwned<Tensor>, 1> outputs_;
};
Tensor wrapper_ne_Tensor(const Tensor & self, const Tensor & other) {
    structured_ne_Tensor_out_functional op;
    op.meta(self, other);
    op.impl(self, other, *op.outputs_[0]);
    return std::move(op.outputs_[0]).take();
}
struct structured_ne_Tensor_out_out final : public structured_ne_Tensor_out {
    structured_ne_Tensor_out_out(Tensor& out0) : outputs_{ std::ref(out0) } {}
    void set_output(int64_t output_idx, IntArrayRef sizes, IntArrayRef strides,
                    TensorOptions options) override {
        const auto& out = outputs_[output_idx].get();
        resize_out(out, sizes, strides, options);
        
        structured_ne_Tensor_out::set_output(output_idx, sizes, strides, options);
    }
    const Tensor& maybe_get_output(int64_t output_idx) override {
        return outputs_[output_idx];
    }
    std::array<std::reference_wrapper<Tensor>, 1> outputs_;
};
Tensor & wrapper_ne_out_Tensor_out(const Tensor & self, const Tensor & other, Tensor & out) {
    structured_ne_Tensor_out_out op(out);
    op.meta(self, other);
    op.impl(self, other, op.outputs_[0]);
    return out;
}
struct structured_ne_Tensor_out_inplace final : public structured_ne_Tensor_out {
    structured_ne_Tensor_out_inplace(Tensor& self) : outputs_{std::ref(self)} {}
    void set_output(int64_t output_idx, IntArrayRef sizes, IntArrayRef strides,
                    TensorOptions options) override {
        const auto& out = outputs_[output_idx].get();
        check_inplace(out, sizes, options);
        
        structured_ne_Tensor_out::set_output(output_idx, sizes, strides, options);
    }
    const Tensor& maybe_get_output(int64_t output_idx) override {
        return outputs_[output_idx];
    }
    std::array<std::reference_wrapper<Tensor>, 1> outputs_;
};
Tensor & wrapper_ne__Tensor(Tensor & self, const Tensor & other) {
    structured_ne_Tensor_out_inplace op(self);
    op.meta(self, other);
    op.impl(self, other, op.outputs_[0]);
    return self;
}
struct structured_ge_Scalar_out_functional final : public structured_ge_Scalar_out {
    void set_output(int64_t output_idx, IntArrayRef sizes, IntArrayRef strides,
                    TensorOptions options) override {
        outputs_[output_idx] = create_out(sizes, strides, options);
        
        structured_ge_Scalar_out::set_output(output_idx, sizes, strides, options);
    }
    const Tensor& maybe_get_output(int64_t output_idx) override {
        return *outputs_[output_idx];
    }
    std::array<ExclusivelyOwned<Tensor>, 1> outputs_;
};
Tensor wrapper_ge_Scalar(const Tensor & self, const Scalar & other) {
    structured_ge_Scalar_out_functional op;
    op.meta(self, other);
    op.impl(self, other, *op.outputs_[0]);
    return std::move(op.outputs_[0]).take();
}
struct structured_ge_Scalar_out_out final : public structured_ge_Scalar_out {
    structured_ge_Scalar_out_out(Tensor& out0) : outputs_{ std::ref(out0) } {}
    void set_output(int64_t output_idx, IntArrayRef sizes, IntArrayRef strides,
                    TensorOptions options) override {
        const auto& out = outputs_[output_idx].get();
        resize_out(out, sizes, strides, options);
        
        structured_ge_Scalar_out::set_output(output_idx, sizes, strides, options);
    }
    const Tensor& maybe_get_output(int64_t output_idx) override {
        return outputs_[output_idx];
    }
    std::array<std::reference_wrapper<Tensor>, 1> outputs_;
};
Tensor & wrapper_ge_out_Scalar_out(const Tensor & self, const Scalar & other, Tensor & out) {
    structured_ge_Scalar_out_out op(out);
    op.meta(self, other);
    op.impl(self, other, op.outputs_[0]);
    return out;
}
struct structured_ge_Scalar_out_inplace final : public structured_ge_Scalar_out {
    structured_ge_Scalar_out_inplace(Tensor& self) : outputs_{std::ref(self)} {}
    void set_output(int64_t output_idx, IntArrayRef sizes, IntArrayRef strides,
                    TensorOptions options) override {
        const auto& out = outputs_[output_idx].get();
        check_inplace(out, sizes, options);
        
        structured_ge_Scalar_out::set_output(output_idx, sizes, strides, options);
    }
    const Tensor& maybe_get_output(int64_t output_idx) override {
        return outputs_[output_idx];
    }
    std::array<std::reference_wrapper<Tensor>, 1> outputs_;
};
Tensor & wrapper_ge__Scalar(Tensor & self, const Scalar & other) {
    structured_ge_Scalar_out_inplace op(self);
    op.meta(self, other);
    op.impl(self, other, op.outputs_[0]);
    return self;
}
struct structured_ge_Tensor_out_functional final : public structured_ge_Tensor_out {
    void set_output(int64_t output_idx, IntArrayRef sizes, IntArrayRef strides,
                    TensorOptions options) override {
        outputs_[output_idx] = create_out(sizes, strides, options);
        
        structured_ge_Tensor_out::set_output(output_idx, sizes, strides, options);
    }
    const Tensor& maybe_get_output(int64_t output_idx) override {
        return *outputs_[output_idx];
    }
    std::array<ExclusivelyOwned<Tensor>, 1> outputs_;
};
Tensor wrapper_ge_Tensor(const Tensor & self, const Tensor & other) {
    structured_ge_Tensor_out_functional op;
    op.meta(self, other);
    op.impl(self, other, *op.outputs_[0]);
    return std::move(op.outputs_[0]).take();
}
struct structured_ge_Tensor_out_out final : public structured_ge_Tensor_out {
    structured_ge_Tensor_out_out(Tensor& out0) : outputs_{ std::ref(out0) } {}
    void set_output(int64_t output_idx, IntArrayRef sizes, IntArrayRef strides,
                    TensorOptions options) override {
        const auto& out = outputs_[output_idx].get();
        resize_out(out, sizes, strides, options);
        
        structured_ge_Tensor_out::set_output(output_idx, sizes, strides, options);
    }
    const Tensor& maybe_get_output(int64_t output_idx) override {
        return outputs_[output_idx];
    }
    std::array<std::reference_wrapper<Tensor>, 1> outputs_;
};
Tensor & wrapper_ge_out_Tensor_out(const Tensor & self, const Tensor & other, Tensor & out) {
    structured_ge_Tensor_out_out op(out);
    op.meta(self, other);
    op.impl(self, other, op.outputs_[0]);
    return out;
}
struct structured_ge_Tensor_out_inplace final : public structured_ge_Tensor_out {
    structured_ge_Tensor_out_inplace(Tensor& self) : outputs_{std::ref(self)} {}
    void set_output(int64_t output_idx, IntArrayRef sizes, IntArrayRef strides,
                    TensorOptions options) override {
        const auto& out = outputs_[output_idx].get();
        check_inplace(out, sizes, options);
        
        structured_ge_Tensor_out::set_output(output_idx, sizes, strides, options);
    }
    const Tensor& maybe_get_output(int64_t output_idx) override {
        return outputs_[output_idx];
    }
    std::array<std::reference_wrapper<Tensor>, 1> outputs_;
};
Tensor & wrapper_ge__Tensor(Tensor & self, const Tensor & other) {
    structured_ge_Tensor_out_inplace op(self);
    op.meta(self, other);
    op.impl(self, other, op.outputs_[0]);
    return self;
}
struct structured_le_Scalar_out_functional final : public structured_le_Scalar_out {
    void set_output(int64_t output_idx, IntArrayRef sizes, IntArrayRef strides,
                    TensorOptions options) override {
        outputs_[output_idx] = create_out(sizes, strides, options);
        
        structured_le_Scalar_out::set_output(output_idx, sizes, strides, options);
    }
    const Tensor& maybe_get_output(int64_t output_idx) override {
        return *outputs_[output_idx];
    }
    std::array<ExclusivelyOwned<Tensor>, 1> outputs_;
};
Tensor wrapper_le_Scalar(const Tensor & self, const Scalar & other) {
    structured_le_Scalar_out_functional op;
    op.meta(self, other);
    op.impl(self, other, *op.outputs_[0]);
    return std::move(op.outputs_[0]).take();
}
struct structured_le_Scalar_out_out final : public structured_le_Scalar_out {
    structured_le_Scalar_out_out(Tensor& out0) : outputs_{ std::ref(out0) } {}
    void set_output(int64_t output_idx, IntArrayRef sizes, IntArrayRef strides,
                    TensorOptions options) override {
        const auto& out = outputs_[output_idx].get();
        resize_out(out, sizes, strides, options);
        
        structured_le_Scalar_out::set_output(output_idx, sizes, strides, options);
    }
    const Tensor& maybe_get_output(int64_t output_idx) override {
        return outputs_[output_idx];
    }
    std::array<std::reference_wrapper<Tensor>, 1> outputs_;
};
Tensor & wrapper_le_out_Scalar_out(const Tensor & self, const Scalar & other, Tensor & out) {
    structured_le_Scalar_out_out op(out);
    op.meta(self, other);
    op.impl(self, other, op.outputs_[0]);
    return out;
}
struct structured_le_Scalar_out_inplace final : public structured_le_Scalar_out {
    structured_le_Scalar_out_inplace(Tensor& self) : outputs_{std::ref(self)} {}
    void set_output(int64_t output_idx, IntArrayRef sizes, IntArrayRef strides,
                    TensorOptions options) override {
        const auto& out = outputs_[output_idx].get();
        check_inplace(out, sizes, options);
        
        structured_le_Scalar_out::set_output(output_idx, sizes, strides, options);
    }
    const Tensor& maybe_get_output(int64_t output_idx) override {
        return outputs_[output_idx];
    }
    std::array<std::reference_wrapper<Tensor>, 1> outputs_;
};
Tensor & wrapper_le__Scalar(Tensor & self, const Scalar & other) {
    structured_le_Scalar_out_inplace op(self);
    op.meta(self, other);
    op.impl(self, other, op.outputs_[0]);
    return self;
}
struct structured_le_Tensor_out_functional final : public structured_le_Tensor_out {
    void set_output(int64_t output_idx, IntArrayRef sizes, IntArrayRef strides,
                    TensorOptions options) override {
        outputs_[output_idx] = create_out(sizes, strides, options);
        
        structured_le_Tensor_out::set_output(output_idx, sizes, strides, options);
    }
    const Tensor& maybe_get_output(int64_t output_idx) override {
        return *outputs_[output_idx];
    }
    std::array<ExclusivelyOwned<Tensor>, 1> outputs_;
};
Tensor wrapper_le_Tensor(const Tensor & self, const Tensor & other) {
    structured_le_Tensor_out_functional op;
    op.meta(self, other);
    op.impl(self, other, *op.outputs_[0]);
    return std::move(op.outputs_[0]).take();
}
struct structured_le_Tensor_out_out final : public structured_le_Tensor_out {
    structured_le_Tensor_out_out(Tensor& out0) : outputs_{ std::ref(out0) } {}
    void set_output(int64_t output_idx, IntArrayRef sizes, IntArrayRef strides,
                    TensorOptions options) override {
        const auto& out = outputs_[output_idx].get();
        resize_out(out, sizes, strides, options);
        
        structured_le_Tensor_out::set_output(output_idx, sizes, strides, options);
    }
    const Tensor& maybe_get_output(int64_t output_idx) override {
        return outputs_[output_idx];
    }
    std::array<std::reference_wrapper<Tensor>, 1> outputs_;
};
Tensor & wrapper_le_out_Tensor_out(const Tensor & self, const Tensor & other, Tensor & out) {
    structured_le_Tensor_out_out op(out);
    op.meta(self, other);
    op.impl(self, other, op.outputs_[0]);
    return out;
}
struct structured_le_Tensor_out_inplace final : public structured_le_Tensor_out {
    structured_le_Tensor_out_inplace(Tensor& self) : outputs_{std::ref(self)} {}
    void set_output(int64_t output_idx, IntArrayRef sizes, IntArrayRef strides,
                    TensorOptions options) override {
        const auto& out = outputs_[output_idx].get();
        check_inplace(out, sizes, options);
        
        structured_le_Tensor_out::set_output(output_idx, sizes, strides, options);
    }
    const Tensor& maybe_get_output(int64_t output_idx) override {
        return outputs_[output_idx];
    }
    std::array<std::reference_wrapper<Tensor>, 1> outputs_;
};
Tensor & wrapper_le__Tensor(Tensor & self, const Tensor & other) {
    structured_le_Tensor_out_inplace op(self);
    op.meta(self, other);
    op.impl(self, other, op.outputs_[0]);
    return self;
}
struct structured_gt_Scalar_out_functional final : public structured_gt_Scalar_out {
    void set_output(int64_t output_idx, IntArrayRef sizes, IntArrayRef strides,
                    TensorOptions options) override {
        outputs_[output_idx] = create_out(sizes, strides, options);
        
        structured_gt_Scalar_out::set_output(output_idx, sizes, strides, options);
    }
    const Tensor& maybe_get_output(int64_t output_idx) override {
        return *outputs_[output_idx];
    }
    std::array<ExclusivelyOwned<Tensor>, 1> outputs_;
};
Tensor wrapper_gt_Scalar(const Tensor & self, const Scalar & other) {
    structured_gt_Scalar_out_functional op;
    op.meta(self, other);
    op.impl(self, other, *op.outputs_[0]);
    return std::move(op.outputs_[0]).take();
}
struct structured_gt_Scalar_out_out final : public structured_gt_Scalar_out {
    structured_gt_Scalar_out_out(Tensor& out0) : outputs_{ std::ref(out0) } {}
    void set_output(int64_t output_idx, IntArrayRef sizes, IntArrayRef strides,
                    TensorOptions options) override {
        const auto& out = outputs_[output_idx].get();
        resize_out(out, sizes, strides, options);
        
        structured_gt_Scalar_out::set_output(output_idx, sizes, strides, options);
    }
    const Tensor& maybe_get_output(int64_t output_idx) override {
        return outputs_[output_idx];
    }
    std::array<std::reference_wrapper<Tensor>, 1> outputs_;
};
Tensor & wrapper_gt_out_Scalar_out(const Tensor & self, const Scalar & other, Tensor & out) {
    structured_gt_Scalar_out_out op(out);
    op.meta(self, other);
    op.impl(self, other, op.outputs_[0]);
    return out;
}
struct structured_gt_Scalar_out_inplace final : public structured_gt_Scalar_out {
    structured_gt_Scalar_out_inplace(Tensor& self) : outputs_{std::ref(self)} {}
    void set_output(int64_t output_idx, IntArrayRef sizes, IntArrayRef strides,
                    TensorOptions options) override {
        const auto& out = outputs_[output_idx].get();
        check_inplace(out, sizes, options);
        
        structured_gt_Scalar_out::set_output(output_idx, sizes, strides, options);
    }
    const Tensor& maybe_get_output(int64_t output_idx) override {
        return outputs_[output_idx];
    }
    std::array<std::reference_wrapper<Tensor>, 1> outputs_;
};
Tensor & wrapper_gt__Scalar(Tensor & self, const Scalar & other) {
    structured_gt_Scalar_out_inplace op(self);
    op.meta(self, other);
    op.impl(self, other, op.outputs_[0]);
    return self;
}
struct structured_gt_Tensor_out_functional final : public structured_gt_Tensor_out {
    void set_output(int64_t output_idx, IntArrayRef sizes, IntArrayRef strides,
                    TensorOptions options) override {
        outputs_[output_idx] = create_out(sizes, strides, options);
        
        structured_gt_Tensor_out::set_output(output_idx, sizes, strides, options);
    }
    const Tensor& maybe_get_output(int64_t output_idx) override {
        return *outputs_[output_idx];
    }
    std::array<ExclusivelyOwned<Tensor>, 1> outputs_;
};
Tensor wrapper_gt_Tensor(const Tensor & self, const Tensor & other) {
    structured_gt_Tensor_out_functional op;
    op.meta(self, other);
    op.impl(self, other, *op.outputs_[0]);
    return std::move(op.outputs_[0]).take();
}
struct structured_gt_Tensor_out_out final : public structured_gt_Tensor_out {
    structured_gt_Tensor_out_out(Tensor& out0) : outputs_{ std::ref(out0) } {}
    void set_output(int64_t output_idx, IntArrayRef sizes, IntArrayRef strides,
                    TensorOptions options) override {
        const auto& out = outputs_[output_idx].get();
        resize_out(out, sizes, strides, options);
        
        structured_gt_Tensor_out::set_output(output_idx, sizes, strides, options);
    }
    const Tensor& maybe_get_output(int64_t output_idx) override {
        return outputs_[output_idx];
    }
    std::array<std::reference_wrapper<Tensor>, 1> outputs_;
};
Tensor & wrapper_gt_out_Tensor_out(const Tensor & self, const Tensor & other, Tensor & out) {
    structured_gt_Tensor_out_out op(out);
    op.meta(self, other);
    op.impl(self, other, op.outputs_[0]);
    return out;
}
struct structured_gt_Tensor_out_inplace final : public structured_gt_Tensor_out {
    structured_gt_Tensor_out_inplace(Tensor& self) : outputs_{std::ref(self)} {}
    void set_output(int64_t output_idx, IntArrayRef sizes, IntArrayRef strides,
                    TensorOptions options) override {
        const auto& out = outputs_[output_idx].get();
        check_inplace(out, sizes, options);
        
        structured_gt_Tensor_out::set_output(output_idx, sizes, strides, options);
    }
    const Tensor& maybe_get_output(int64_t output_idx) override {
        return outputs_[output_idx];
    }
    std::array<std::reference_wrapper<Tensor>, 1> outputs_;
};
Tensor & wrapper_gt__Tensor(Tensor & self, const Tensor & other) {
    structured_gt_Tensor_out_inplace op(self);
    op.meta(self, other);
    op.impl(self, other, op.outputs_[0]);
    return self;
}
struct structured_lt_Scalar_out_functional final : public structured_lt_Scalar_out {
    void set_output(int64_t output_idx, IntArrayRef sizes, IntArrayRef strides,
                    TensorOptions options) override {
        outputs_[output_idx] = create_out(sizes, strides, options);
        
        structured_lt_Scalar_out::set_output(output_idx, sizes, strides, options);
    }
    const Tensor& maybe_get_output(int64_t output_idx) override {
        return *outputs_[output_idx];
    }
    std::array<ExclusivelyOwned<Tensor>, 1> outputs_;
};
Tensor wrapper_lt_Scalar(const Tensor & self, const Scalar & other) {
    structured_lt_Scalar_out_functional op;
    op.meta(self, other);
    op.impl(self, other, *op.outputs_[0]);
    return std::move(op.outputs_[0]).take();
}
struct structured_lt_Scalar_out_out final : public structured_lt_Scalar_out {
    structured_lt_Scalar_out_out(Tensor& out0) : outputs_{ std::ref(out0) } {}
    void set_output(int64_t output_idx, IntArrayRef sizes, IntArrayRef strides,
                    TensorOptions options) override {
        const auto& out = outputs_[output_idx].get();
        resize_out(out, sizes, strides, options);
        
        structured_lt_Scalar_out::set_output(output_idx, sizes, strides, options);
    }
    const Tensor& maybe_get_output(int64_t output_idx) override {
        return outputs_[output_idx];
    }
    std::array<std::reference_wrapper<Tensor>, 1> outputs_;
};
Tensor & wrapper_lt_out_Scalar_out(const Tensor & self, const Scalar & other, Tensor & out) {
    structured_lt_Scalar_out_out op(out);
    op.meta(self, other);
    op.impl(self, other, op.outputs_[0]);
    return out;
}
struct structured_lt_Scalar_out_inplace final : public structured_lt_Scalar_out {
    structured_lt_Scalar_out_inplace(Tensor& self) : outputs_{std::ref(self)} {}
    void set_output(int64_t output_idx, IntArrayRef sizes, IntArrayRef strides,
                    TensorOptions options) override {
        const auto& out = outputs_[output_idx].get();
        check_inplace(out, sizes, options);
        
        structured_lt_Scalar_out::set_output(output_idx, sizes, strides, options);
    }
    const Tensor& maybe_get_output(int64_t output_idx) override {
        return outputs_[output_idx];
    }
    std::array<std::reference_wrapper<Tensor>, 1> outputs_;
};
Tensor & wrapper_lt__Scalar(Tensor & self, const Scalar & other) {
    structured_lt_Scalar_out_inplace op(self);
    op.meta(self, other);
    op.impl(self, other, op.outputs_[0]);
    return self;
}
struct structured_lt_Tensor_out_functional final : public structured_lt_Tensor_out {
    void set_output(int64_t output_idx, IntArrayRef sizes, IntArrayRef strides,
                    TensorOptions options) override {
        outputs_[output_idx] = create_out(sizes, strides, options);
        
        structured_lt_Tensor_out::set_output(output_idx, sizes, strides, options);
    }
    const Tensor& maybe_get_output(int64_t output_idx) override {
        return *outputs_[output_idx];
    }
    std::array<ExclusivelyOwned<Tensor>, 1> outputs_;
};
Tensor wrapper_lt_Tensor(const Tensor & self, const Tensor & other) {
    structured_lt_Tensor_out_functional op;
    op.meta(self, other);
    op.impl(self, other, *op.outputs_[0]);
    return std::move(op.outputs_[0]).take();
}
struct structured_lt_Tensor_out_out final : public structured_lt_Tensor_out {
    structured_lt_Tensor_out_out(Tensor& out0) : outputs_{ std::ref(out0) } {}
    void set_output(int64_t output_idx, IntArrayRef sizes, IntArrayRef strides,
                    TensorOptions options) override {
        const auto& out = outputs_[output_idx].get();
        resize_out(out, sizes, strides, options);
        
        structured_lt_Tensor_out::set_output(output_idx, sizes, strides, options);
    }
    const Tensor& maybe_get_output(int64_t output_idx) override {
        return outputs_[output_idx];
    }
    std::array<std::reference_wrapper<Tensor>, 1> outputs_;
};
Tensor & wrapper_lt_out_Tensor_out(const Tensor & self, const Tensor & other, Tensor & out) {
    structured_lt_Tensor_out_out op(out);
    op.meta(self, other);
    op.impl(self, other, op.outputs_[0]);
    return out;
}
struct structured_lt_Tensor_out_inplace final : public structured_lt_Tensor_out {
    structured_lt_Tensor_out_inplace(Tensor& self) : outputs_{std::ref(self)} {}
    void set_output(int64_t output_idx, IntArrayRef sizes, IntArrayRef strides,
                    TensorOptions options) override {
        const auto& out = outputs_[output_idx].get();
        check_inplace(out, sizes, options);
        
        structured_lt_Tensor_out::set_output(output_idx, sizes, strides, options);
    }
    const Tensor& maybe_get_output(int64_t output_idx) override {
        return outputs_[output_idx];
    }
    std::array<std::reference_wrapper<Tensor>, 1> outputs_;
};
Tensor & wrapper_lt__Tensor(Tensor & self, const Tensor & other) {
    structured_lt_Tensor_out_inplace op(self);
    op.meta(self, other);
    op.impl(self, other, op.outputs_[0]);
    return self;
}

struct structured_softmax_cpu_out_functional final : public structured_softmax_cpu_out {
    void set_output(int64_t output_idx, IntArrayRef sizes, IntArrayRef strides, TensorOptions options) override {
        outputs_[output_idx] = create_out(sizes, strides, options);
    }
    
    const Tensor& maybe_get_output(int64_t output_idx) override {
        return *outputs_[output_idx];
        
    }
    std::array<ExclusivelyOwned<Tensor>, 1> outputs_;
};

Tensor wrapper__softmax(const Tensor & self, int64_t dim, bool half_to_float) {
    structured_softmax_cpu_out_functional op;
    op.meta(self, dim, half_to_float);
    op.impl(self, dim, half_to_float, *op.outputs_[0]);
    return std::move(op.outputs_[0]).take();
}

struct structured_softmax_cpu_out_out final : public structured_softmax_cpu_out {
    structured_softmax_cpu_out_out(Tensor& out0) : outputs_{ std::ref(out0) } {}
    
    void set_output(int64_t output_idx, IntArrayRef sizes, IntArrayRef strides, TensorOptions options) override {
        const auto& out = outputs_[output_idx].get();
        resize_out(out, sizes, strides, options);
    }
    
    const Tensor& maybe_get_output(int64_t output_idx) override {
        return outputs_[output_idx];
    }
    
    std::array<std::reference_wrapper<Tensor>, 1> outputs_;
};

Tensor & wrapper__softmax_out_out(const Tensor & self, int64_t dim, bool half_to_float, Tensor & out) {
    structured_softmax_cpu_out_out op(out);
    op.meta(self, dim, half_to_float);
    op.impl(self, dim, half_to_float, op.outputs_[0]);
    return out;
}

struct structured_topk_out_cpu_functional final : public structured_topk_out_cpu {
    void set_output(int64_t output_idx, IntArrayRef sizes, IntArrayRef strides, TensorOptions options) override {
        
        outputs_[output_idx] = create_out(sizes, strides, options);
    }
    
    const Tensor& maybe_get_output(int64_t output_idx) override {
        return *outputs_[output_idx];
        
    }
    std::array<otter::ExclusivelyOwned<Tensor>, 2> outputs_;
};

::std::tuple<Tensor, Tensor> wrapper_topk(const Tensor & self, int64_t k, int64_t dim, bool largest, bool sorted) {
    structured_topk_out_cpu_functional op;
    op.meta(self, k, dim, largest, sorted);
    op.impl(self, k, dim, largest, sorted, *op.outputs_[0], *op.outputs_[1]);
    return std::make_tuple(std::move(op.outputs_[0]).take(), std::move(op.outputs_[1]).take());
}
struct structured_topk_out_cpu_out final : public structured_topk_out_cpu {
    structured_topk_out_cpu_out(Tensor& out0, Tensor& out1) : outputs_{ std::ref(out0), std::ref(out1) } {}
    
    
    void set_output(int64_t output_idx, IntArrayRef sizes, IntArrayRef strides, TensorOptions options) override {
        
        const auto& out = outputs_[output_idx].get();
        resize_out(out, sizes, strides, options);
    }
    
    const Tensor& maybe_get_output(int64_t output_idx) override {
        return outputs_[output_idx];
        
    }
    std::array<std::reference_wrapper<Tensor>, 2> outputs_;
};

::std::tuple<Tensor &, Tensor &> wrapper_topk_out_values(const Tensor & self, int64_t k, int64_t dim, bool largest, bool sorted, Tensor & values, Tensor & indices) {
    structured_topk_out_cpu_out op(values, indices);
    op.meta(self, k, dim, largest, sorted);
    Tensor values_in = op.maybe_get_output(0);
    Tensor indices_in = op.maybe_get_output(1);
    op.impl(self, k, dim, largest, sorted, values_in, indices_in);
    return std::forward_as_tuple(values, indices);
}

struct structured_sort_stable_out_functional final : public structured_sort_stable_out {
    void set_output(int64_t output_idx, IntArrayRef sizes, IntArrayRef strides, TensorOptions options) override {
        outputs_[output_idx] = create_out(sizes, strides, options);
    }
    
    const Tensor& maybe_get_output(int64_t output_idx) override {
        return *outputs_[output_idx];
    }
    
    std::array<otter::ExclusivelyOwned<Tensor>, 2> outputs_;
};
::std::tuple<Tensor, Tensor> wrapper_sort_stable(const Tensor & self, bool stable, int64_t dim, bool descending) {
    structured_sort_stable_out_functional op;
    op.meta(self, stable, dim, descending);
    op.impl(self, stable, dim, descending, *op.outputs_[0], *op.outputs_[1]);
    return std::make_tuple(std::move(op.outputs_[0]).take(), std::move(op.outputs_[1]).take());
}
struct structured_sort_stable_out_out final : public structured_sort_stable_out {
    structured_sort_stable_out_out(Tensor& out0, Tensor& out1) : outputs_{ std::ref(out0), std::ref(out1) } {}
    
    void set_output(int64_t output_idx, IntArrayRef sizes, IntArrayRef strides, TensorOptions options) override {
        const auto& out = outputs_[output_idx].get();
        resize_out(out, sizes, strides, options);
    }
    
    const Tensor& maybe_get_output(int64_t output_idx) override {
        return outputs_[output_idx];
    }
    std::array<std::reference_wrapper<Tensor>, 2> outputs_;
};

::std::tuple<Tensor &, Tensor &> wrapper_sort_out_values_stable(const Tensor & self, bool stable, int64_t dim, bool descending, Tensor & values, Tensor & indices) {
    structured_sort_stable_out_out op(values, indices);
    Tensor values_in = op.maybe_get_output(0);
    Tensor indices_in = op.maybe_get_output(1);
    op.meta(self, stable, dim, descending);
    op.impl(self, stable, dim, descending, values_in, indices_in);
    return std::forward_as_tuple(values, indices);
}

struct structured_gather_out_functional final : public structured_gather_out {
    void set_output(int64_t output_idx, IntArrayRef sizes, IntArrayRef strides, TensorOptions options) override {
        outputs_[output_idx] = create_out(sizes, strides, options);
    }
    
    const Tensor& maybe_get_output(int64_t output_idx) override {
        return *outputs_[output_idx];
    }
    std::array<otter::ExclusivelyOwned<Tensor>, 1> outputs_;
};
Tensor wrapper_gather(const Tensor & self, int64_t dim, const Tensor & index, bool sparse_grad) {
    structured_gather_out_functional op;
    op.meta(self, dim, index, sparse_grad);
    op.impl(self, dim, index, sparse_grad, *op.outputs_[0]);
    return std::move(op.outputs_[0]).take();
}
struct structured_gather_out_out final : public structured_gather_out {
    structured_gather_out_out(Tensor& out0) : outputs_{ std::ref(out0) } {}
    void set_output(int64_t output_idx, IntArrayRef sizes, IntArrayRef strides, TensorOptions options) override {
        const auto& out = outputs_[output_idx].get();
        resize_out(out, sizes, strides, options);
        
    }
    
    const Tensor& maybe_get_output(int64_t output_idx) override {
        return outputs_[output_idx];
    }
    std::array<std::reference_wrapper<Tensor>, 1> outputs_;
    
};
Tensor & wrapper_gather_out_out(const Tensor & self, int64_t dim, const Tensor & index, bool sparse_grad, Tensor & out) {
    structured_gather_out_out op(out);
    op.meta(self, dim, index, sparse_grad);
    op.impl(self, dim, index, sparse_grad, op.maybe_get_output(0));
    return out;
}

struct structured_scatter_src_out_functional final : public structured_scatter_src_out {
    void set_output(int64_t output_idx, IntArrayRef sizes, IntArrayRef strides, TensorOptions options) override {
        outputs_[output_idx] = create_out(sizes, strides, options);
        
    }
    const Tensor& maybe_get_output(int64_t output_idx) override {
        return *outputs_[output_idx];
    }
    std::array<otter::ExclusivelyOwned<Tensor>, 1> outputs_;
};
Tensor wrapper_scatter_src(const Tensor & self, int64_t dim, const Tensor & index, const Tensor & src) {
    structured_scatter_src_out_functional op;
    op.meta(self, dim, index, src);
    op.impl(self, dim, index, src, *op.outputs_[0]);
    return std::move(op.outputs_[0]).take();
}
struct structured_scatter_src_out_out final : public structured_scatter_src_out {
    structured_scatter_src_out_out(Tensor& out0) : outputs_{ std::ref(out0) } {}
    void set_output(int64_t output_idx, IntArrayRef sizes, IntArrayRef strides, TensorOptions options) override {
        const auto& out = outputs_[output_idx].get();
        resize_out(out, sizes, strides, options);
        
    }
    const Tensor& maybe_get_output(int64_t output_idx) override {
        return outputs_[output_idx];
    }
    std::array<std::reference_wrapper<Tensor>, 1> outputs_;
    
};
Tensor & wrapper_scatter_out_src_out(const Tensor & self, int64_t dim, const Tensor & index, const Tensor & src, Tensor & out) {
    structured_scatter_src_out_out op(out);
    op.meta(self, dim, index, src);
    op.impl(self, dim, index, src, op.maybe_get_output(0));
    
    return out;
}
struct structured_scatter_src_out_inplace final : public structured_scatter_src_out {
    structured_scatter_src_out_inplace(Tensor& self) : outputs_{std::ref(self)} {}
    void set_output(int64_t output_idx, IntArrayRef sizes, IntArrayRef /*strides*/, TensorOptions options) override {
        const auto& out = outputs_[output_idx].get();
        check_inplace(out, sizes, options);
        
    }
    const Tensor& maybe_get_output(int64_t output_idx) override {
        return outputs_[output_idx];
    }
    std::array<std::reference_wrapper<Tensor>, 1> outputs_;
    
};
Tensor & wrapper_scatter__src(Tensor & self, int64_t dim, const Tensor & index, const Tensor & src) {
    structured_scatter_src_out_inplace op(self);
    op.meta(self, dim, index, src);
    op.impl(self, dim, index, src, op.outputs_[0]);
    
    return self;
}
struct structured_scatter_value_out_functional final : public structured_scatter_value_out {
    void set_output(int64_t output_idx, IntArrayRef sizes, IntArrayRef strides, TensorOptions options) override {
        outputs_[output_idx] = create_out(sizes, strides, options);
        
    }
    const Tensor& maybe_get_output(int64_t output_idx) override {
        return *outputs_[output_idx];
    }
    std::array<otter::ExclusivelyOwned<Tensor>, 1> outputs_;
};
Tensor wrapper_scatter_value(const Tensor & self, int64_t dim, const Tensor & index, const Scalar & value) {
    structured_scatter_value_out_functional op;
    op.meta(self, dim, index, value);
    op.impl(self, dim, index, value, *op.outputs_[0]);
    return std::move(op.outputs_[0]).take();
}
struct structured_scatter_value_out_out final : public structured_scatter_value_out {
    structured_scatter_value_out_out(Tensor& out0) : outputs_{ std::ref(out0) } {}
    void set_output(int64_t output_idx, IntArrayRef sizes, IntArrayRef strides, TensorOptions options) override {
        const auto& out = outputs_[output_idx].get();
        resize_out(out, sizes, strides, options);
        
    }
    const Tensor& maybe_get_output(int64_t output_idx) override {
        return outputs_[output_idx];
    }
    std::array<std::reference_wrapper<Tensor>, 1> outputs_;
    
};
Tensor & wrapper_scatter_out_value_out(const Tensor & self, int64_t dim, const Tensor & index, const Scalar & value, Tensor & out) {
    structured_scatter_value_out_out op(out);
    op.meta(self, dim, index, value);
    op.impl(self, dim, index, value, op.maybe_get_output(0));
    
    return out;
}
struct structured_scatter_value_out_inplace final : public structured_scatter_value_out {
    structured_scatter_value_out_inplace(Tensor& self) : outputs_{std::ref(self)} {}
    void set_output(int64_t output_idx, IntArrayRef sizes, IntArrayRef /*strides*/, TensorOptions options) override {
        const auto& out = outputs_[output_idx].get();
        check_inplace(out, sizes, options);
        
    }
    const Tensor& maybe_get_output(int64_t output_idx) override {
        return outputs_[output_idx];
    }
    std::array<std::reference_wrapper<Tensor>, 1> outputs_;
    
};
Tensor & wrapper_scatter__value(Tensor & self, int64_t dim, const Tensor & index, const Scalar & value) {
    structured_scatter_value_out_inplace op(self);
    op.meta(self, dim, index, value);
    op.impl(self, dim, index, value, op.outputs_[0]);
    
    return self;
}
struct structured_scatter_reduce_out_functional final : public structured_scatter_reduce_out {
    void set_output(int64_t output_idx, IntArrayRef sizes, IntArrayRef strides, TensorOptions options) override {
        outputs_[output_idx] = create_out(sizes, strides, options);
        
    }
    const Tensor& maybe_get_output(int64_t output_idx) override {
        return *outputs_[output_idx];
    }
    std::array<otter::ExclusivelyOwned<Tensor>, 1> outputs_;
};
Tensor wrapper_scatter_reduce(const Tensor & self, int64_t dim, const Tensor & index, const Tensor & src, int64_t reduce) {
    structured_scatter_reduce_out_functional op;
    op.meta(self, dim, index, src, reduce);
    op.impl(self, dim, index, src, reduce, *op.outputs_[0]);
    return std::move(op.outputs_[0]).take();
}
struct structured_scatter_reduce_out_out final : public structured_scatter_reduce_out {
    structured_scatter_reduce_out_out(Tensor& out0) : outputs_{ std::ref(out0) } {}
    void set_output(int64_t output_idx, IntArrayRef sizes, IntArrayRef strides, TensorOptions options) override {
        const auto& out = outputs_[output_idx].get();
        resize_out(out, sizes, strides, options);
        
    }
    const Tensor& maybe_get_output(int64_t output_idx) override {
        return outputs_[output_idx];
    }
    std::array<std::reference_wrapper<Tensor>, 1> outputs_;
    
};
Tensor & wrapper_scatter_out_reduce_out(const Tensor & self, int64_t dim, const Tensor & index, const Tensor & src, int64_t reduce, Tensor & out) {
    structured_scatter_reduce_out_out op(out);
    op.meta(self, dim, index, src, reduce);
    op.impl(self, dim, index, src, reduce, op.maybe_get_output(0));
    
    return out;
}
struct structured_scatter_reduce_out_inplace final : public structured_scatter_reduce_out {
    structured_scatter_reduce_out_inplace(Tensor& self) : outputs_{std::ref(self)} {}
    void set_output(int64_t output_idx, IntArrayRef sizes, IntArrayRef /*strides*/, TensorOptions options) override {
        const auto& out = outputs_[output_idx].get();
        check_inplace(out, sizes, options);
        
    }
    const Tensor& maybe_get_output(int64_t output_idx) override {
        return outputs_[output_idx];
    }
    std::array<std::reference_wrapper<Tensor>, 1> outputs_;
    
};
Tensor & wrapper_scatter__reduce(Tensor & self, int64_t dim, const Tensor & index, const Tensor & src, int64_t reduce) {
    structured_scatter_reduce_out_inplace op(self);
    op.meta(self, dim, index, src, reduce);
    op.impl(self, dim, index, src, reduce, op.outputs_[0]);
    
    return self;
}
struct structured_scatter_value_reduce_out_functional final : public structured_scatter_value_reduce_out {
    void set_output(int64_t output_idx, IntArrayRef sizes, IntArrayRef strides, TensorOptions options) override {
        outputs_[output_idx] = create_out(sizes, strides, options);
        
    }
    const Tensor& maybe_get_output(int64_t output_idx) override {
        return *outputs_[output_idx];
    }
    std::array<otter::ExclusivelyOwned<Tensor>, 1> outputs_;
};
Tensor wrapper_scatter_value_reduce(const Tensor & self, int64_t dim, const Tensor & index, const Scalar & value, int64_t reduce) {
    structured_scatter_value_reduce_out_functional op;
    op.meta(self, dim, index, value, reduce);
    op.impl(self, dim, index, value, reduce, *op.outputs_[0]);
    return std::move(op.outputs_[0]).take();
}
struct structured_scatter_value_reduce_out_out final : public structured_scatter_value_reduce_out {
    structured_scatter_value_reduce_out_out(Tensor& out0) : outputs_{ std::ref(out0) } {}
    void set_output(int64_t output_idx, IntArrayRef sizes, IntArrayRef strides, TensorOptions options) override {
        const auto& out = outputs_[output_idx].get();
        resize_out(out, sizes, strides, options);
        
    }
    const Tensor& maybe_get_output(int64_t output_idx) override {
        return outputs_[output_idx];
    }
    std::array<std::reference_wrapper<Tensor>, 1> outputs_;
    
};
Tensor & wrapper_scatter_out_value_reduce_out(const Tensor & self, int64_t dim, const Tensor & index, const Scalar & value, int64_t reduce, Tensor & out) {
    structured_scatter_value_reduce_out_out op(out);
    op.meta(self, dim, index, value, reduce);
    op.impl(self, dim, index, value, reduce, op.maybe_get_output(0));
    
    return out;
}
struct structured_scatter_value_reduce_out_inplace final : public structured_scatter_value_reduce_out {
    structured_scatter_value_reduce_out_inplace(Tensor& self) : outputs_{std::ref(self)} {}
    void set_output(int64_t output_idx, IntArrayRef sizes, IntArrayRef /*strides*/, TensorOptions options) override {
        const auto& out = outputs_[output_idx].get();
        check_inplace(out, sizes, options);
        
    }
    const Tensor& maybe_get_output(int64_t output_idx) override {
        return outputs_[output_idx];
    }
    std::array<std::reference_wrapper<Tensor>, 1> outputs_;
    
};
Tensor & wrapper_scatter__value_reduce(Tensor & self, int64_t dim, const Tensor & index, const Scalar & value, int64_t reduce) {
    structured_scatter_value_reduce_out_inplace op(self);
    op.meta(self, dim, index, value, reduce);
    op.impl(self, dim, index, value, reduce, op.outputs_[0]);
    
    return self;
}
struct structured_baddbmm_out_cpu_functional final : public structured_baddbmm_out_cpu {
    void set_output(int64_t output_idx, IntArrayRef sizes, IntArrayRef strides, TensorOptions options) override {
        outputs_[output_idx] = create_out(sizes, strides, options);
    }

    const Tensor& maybe_get_output(int64_t output_idx) override {
      return *outputs_[output_idx];
    }
    std::array<otter::ExclusivelyOwned<Tensor>, 1> outputs_;
};
Tensor wrapper_baddbmm(const Tensor & self, const Tensor & batch1, const Tensor & batch2, const Scalar & beta, const Scalar & alpha) {
    structured_baddbmm_out_cpu_functional op;
    op.meta(self, batch1, batch2, beta, alpha);
    op.impl(self, batch1, batch2, beta, alpha, *op.outputs_[0]);
    return std::move(op.outputs_[0]).take();
}
struct structured_baddbmm_out_cpu_out final : public structured_baddbmm_out_cpu {
    structured_baddbmm_out_cpu_out(Tensor& out0) : outputs_{ std::ref(out0) } {}
    
    void set_output(int64_t output_idx, IntArrayRef sizes, IntArrayRef strides, TensorOptions options) override {
        const auto& out = outputs_[output_idx].get();
        resize_out(out, sizes, strides, options);

    }

    const Tensor& maybe_get_output(int64_t output_idx) override {
      return outputs_[output_idx];
    }
    std::array<std::reference_wrapper<Tensor>, 1> outputs_;
};
Tensor & wrapper_baddbmm_out_out(const Tensor & self, const Tensor & batch1, const Tensor & batch2, const Scalar & beta, const Scalar & alpha, Tensor & out) {
    structured_baddbmm_out_cpu_out op(out);
    op.meta(self, batch1, batch2, beta, alpha);
    op.impl(self, batch1, batch2, beta, alpha, op.maybe_get_output(0));

    return out;
}
struct structured_baddbmm_out_cpu_inplace final : public structured_baddbmm_out_cpu {
    structured_baddbmm_out_cpu_inplace(Tensor& self) : outputs_{std::ref(self)} {}
    void set_output(int64_t output_idx, IntArrayRef sizes, IntArrayRef /*strides*/, TensorOptions options) override {
        const auto& out = outputs_[output_idx].get();
        check_inplace(out, sizes, options);

    }

    const Tensor& maybe_get_output(int64_t output_idx) override {
      return outputs_[output_idx];
    }
    
    std::array<std::reference_wrapper<Tensor>, 1> outputs_;
};
Tensor & wrapper_baddbmm_(Tensor & self, const Tensor & batch1, const Tensor & batch2, const Scalar & beta, const Scalar & alpha) {
    structured_baddbmm_out_cpu_inplace op(self);
    op.meta(self, batch1, batch2, beta, alpha);
    op.impl(self, batch1, batch2, beta, alpha, op.outputs_[0]);

    return self;
}

struct structured_bmm_out_cpu_functional final : public structured_bmm_out_cpu {
    void set_output(int64_t output_idx, IntArrayRef sizes, IntArrayRef strides, TensorOptions options) override {
        outputs_[output_idx] = create_out(sizes, strides, options);
    }

    const Tensor& maybe_get_output(int64_t output_idx) override {
      return *outputs_[output_idx];
    }
    std::array<otter::ExclusivelyOwned<Tensor>, 1> outputs_;
};
Tensor wrapper_bmm(const Tensor & self, const Tensor & mat2) {
    structured_bmm_out_cpu_functional op;
    op.meta(self, mat2);
    op.impl(self, mat2, *op.outputs_[0]);
    return std::move(op.outputs_[0]).take();
}
struct structured_bmm_out_cpu_out final : public structured_bmm_out_cpu {
    structured_bmm_out_cpu_out(Tensor& out0) : outputs_{ std::ref(out0) } {}
    
    void set_output(int64_t output_idx, IntArrayRef sizes, IntArrayRef strides, TensorOptions options) override {
        const auto& out = outputs_[output_idx].get();
        resize_out(out, sizes, strides, options);

    }

    const Tensor& maybe_get_output(int64_t output_idx) override {
      return outputs_[output_idx];
    }
    std::array<std::reference_wrapper<Tensor>, 1> outputs_;
};
Tensor & wrapper_bmm_out_out(const Tensor & self, const Tensor & mat2, Tensor & out) {
    structured_bmm_out_cpu_out op(out);
    op.meta(self, mat2);
    op.impl(self, mat2, op.maybe_get_output(0));

    return out;
}
struct structured_sum_out_functional final : public structured_sum_out {
    void set_output(int64_t output_idx, IntArrayRef sizes, IntArrayRef strides, TensorOptions options) override {
        outputs_[output_idx] = create_out(sizes, strides, options);
    }
    const Tensor& maybe_get_output(int64_t output_idx) override {
      return *outputs_[output_idx];
    }
    std::array<otter::ExclusivelyOwned<Tensor>, 1> outputs_;
};
Tensor wrapper_sum_dim_IntList(const Tensor & self, IntArrayRef dim, bool keepdim, ScalarType dtype) {
    structured_sum_out_functional op;
    op.meta(self, dim, keepdim, dtype);
    op.impl(self, dim, keepdim, dtype, *op.outputs_[0]);
    return std::move(op.outputs_[0]).take();
}
struct structured_sum_out_out final : public structured_sum_out {
    structured_sum_out_out(Tensor& out0) : outputs_{ std::ref(out0) } {}
    void set_output(int64_t output_idx, IntArrayRef sizes, IntArrayRef strides, TensorOptions options) override {
        const auto& out = outputs_[output_idx].get();
        resize_out(out, sizes, strides, options);
    }
    
    const Tensor& maybe_get_output(int64_t output_idx) override {
      return outputs_[output_idx];
    }
    std::array<std::reference_wrapper<Tensor>, 1> outputs_;
};
Tensor & wrapper_sum_out_IntList_out(const Tensor & self, IntArrayRef dim, bool keepdim, ScalarType dtype, Tensor & out) {
    structured_sum_out_out op(out);
    op.meta(self, dim, keepdim, dtype);
    op.impl(self, dim, keepdim, dtype, op.maybe_get_output(0));
    return out;
}

struct structured_index_out_functional final : public structured_index_out {
    void set_output(int64_t output_idx, IntArrayRef sizes, IntArrayRef strides,TensorOptions options) override {
        outputs_[output_idx] = create_out(sizes, strides, options);

        structured_index_out::set_output(output_idx, sizes, strides, options);
    }
    const Tensor& maybe_get_output(int64_t output_idx) override {
      return *outputs_[output_idx];
    }
    std::array<otter::ExclusivelyOwned<Tensor>, 1> outputs_;
};

Tensor wrapper_index_Tensor(const Tensor & self, const std::vector<otter::optional<Tensor>> & indices) {
    structured_index_out_functional op;
    auto precompute = op.meta(self, indices);
    (void)precompute;
    op.impl(self, precompute.sizes, precompute.strides, *op.outputs_[0]);
    return std::move(op.outputs_[0]).take();
}
struct structured_index_out_out final : public structured_index_out {
    structured_index_out_out(Tensor& out0) : outputs_{ std::ref(out0) } {}
    
    void set_output(int64_t output_idx, IntArrayRef sizes, IntArrayRef strides,TensorOptions options) override {
        const auto& out = outputs_[output_idx].get();
        resize_out(out, sizes, strides, options);

        structured_index_out::set_output(output_idx, sizes, strides, options);
    }
    const Tensor& maybe_get_output(int64_t output_idx) override {
      return outputs_[output_idx];
    }
    std::array<std::reference_wrapper<Tensor>, 1> outputs_;
};
Tensor & wrapper_index_out_Tensor_out(const Tensor & self, const std::vector<otter::optional<Tensor>> & indices, Tensor & out) {
    structured_index_out_out op(out);
    auto precompute = op.meta(self, indices);
    (void)precompute;
    op.impl(self, precompute.sizes, precompute.strides, op.maybe_get_output(0));
    return out;
}

struct structured_prod_out_functional final : public structured_prod_out {
    void set_output(int64_t output_idx, IntArrayRef sizes, IntArrayRef strides, TensorOptions options) override {
        outputs_[output_idx] = create_out(sizes, strides, options);
    }

    const Tensor& maybe_get_output(int64_t output_idx) override {
      return *outputs_[output_idx];
    }
    std::array<ExclusivelyOwned<Tensor>, 1> outputs_;
};
Tensor wrapper_prod_dim_int(const Tensor & self, int64_t dim, bool keepdim, ScalarType dtype) {
    structured_prod_out_functional op;
    op.meta(self, dim, keepdim, dtype);
    op.impl(self, dim, keepdim, dtype, *op.outputs_[0]);
    return std::move(op.outputs_[0]).take();
}
struct structured_prod_out_out final : public structured_prod_out {
    structured_prod_out_out(Tensor& out0) : outputs_{ std::ref(out0) } {}
    void set_output(int64_t output_idx, IntArrayRef sizes, IntArrayRef strides, TensorOptions options) override {
        const auto& out = outputs_[output_idx].get();
        resize_out(out, sizes, strides, options);
    }
    const Tensor& maybe_get_output(int64_t output_idx) override {
      return outputs_[output_idx];
    }
    std::array<std::reference_wrapper<Tensor>, 1> outputs_;
};
Tensor & wrapper_prod_out_int_out(const Tensor & self, int64_t dim, bool keepdim, ScalarType dtype, Tensor & out) {
    structured_prod_out_out op(out);
    op.meta(self, dim, keepdim, dtype);
    op.impl(self, dim, keepdim, dtype, op.maybe_get_output(0));
    return out;
}

struct structured_mean_out_functional final : public structured_mean_out {
    void set_output(int64_t output_idx, IntArrayRef sizes, IntArrayRef strides, TensorOptions options) override {
        outputs_[output_idx] = create_out(sizes, strides, options);
    }

    const Tensor& maybe_get_output(int64_t output_idx) override {
      return *outputs_[output_idx];
    }
    std::array<ExclusivelyOwned<Tensor>, 1> outputs_;
};
Tensor wrapper_mean_dim(const Tensor & self, IntArrayRef dim, bool keepdim, ScalarType dtype) {
    structured_mean_out_functional op;
    op.meta(self, dim, keepdim, dtype);
    op.impl(self, dim, keepdim, dtype, *op.outputs_[0]);
    return std::move(op.outputs_[0]).take();
}
struct structured_mean_out_out final : public structured_mean_out {
    structured_mean_out_out(Tensor& out0) : outputs_{ std::ref(out0) } {}
    void set_output(int64_t output_idx, IntArrayRef sizes, IntArrayRef strides, TensorOptions options) override {
        const auto& out = outputs_[output_idx].get();
        resize_out(out, sizes, strides, options);
    }
    const Tensor& maybe_get_output(int64_t output_idx) override {
        return outputs_[output_idx];
    }
    std::array<std::reference_wrapper<Tensor>, 1> outputs_;
};
Tensor & wrapper_mean_out_out(const Tensor & self, IntArrayRef dim, bool keepdim, ScalarType dtype, Tensor & out) {
    structured_mean_out_out op(out);
    op.meta(self, dim, keepdim, dtype);
    op.impl(self, dim, keepdim, dtype, op.maybe_get_output(0));
    return out;
}

namespace native {

Tensor add(const Tensor & self, const Tensor & other, const Scalar & alpha) {
    return wrapper_add_Tensor(self, other, alpha);
}
Tensor & add_out(Tensor & out, const Tensor & self, const Tensor & other, const Scalar & alpha) {
    return wrapper_add_out_out(self, other, alpha, out);
}
Tensor & add_(Tensor & self, const Tensor & other, const Scalar & alpha) {
    return wrapper_add__Tensor(self, other, alpha);
}

Tensor sub(const Tensor & self, const Tensor & other, const Scalar & alpha) {
    return wrapper_sub_Tensor(self, other, alpha);
}
Tensor & sub_out(Tensor & out, const Tensor & self, const Tensor & other, const Scalar & alpha) {
    return wrapper_sub_out_out(self, other, alpha, out);
}
Tensor & sub_(Tensor & self, const Tensor & other, const Scalar & alpha) {
    return wrapper_sub__Tensor(self, other, alpha);
}

Tensor mul(const Tensor & self, const Tensor & other) {
    return wrapper_mul_Tensor(self, other);
}
Tensor & mul_out(Tensor & out, const Tensor & self, const Tensor & other) {
    return wrapper_mul_out_out(self, other, out);
}
Tensor & mul_(Tensor & self, const Tensor & other) {
    return wrapper_mul__Tensor(self, other);
}

Tensor div(const Tensor & self, const Tensor & other) {
    return wrapper_div_Tensor(self, other);
}
Tensor & div_out(Tensor & out, const Tensor & self, const Tensor & other) {
    return wrapper_div_out_out(self, other, out);
}
Tensor & div_(Tensor & self, const Tensor & other) {
    return wrapper_div__Tensor(self, other);
}

Tensor remainder(const Tensor & self, const Tensor & other) {
    return wrapper_remainder_Tensor(self, other);
}
Tensor & remainder_out(Tensor & out, const Tensor & self, const Tensor & other) {
    return wrapper_remainder_out_out(self, other, out);
}
Tensor & remainder_(Tensor & self, const Tensor & other) {
    return wrapper_remainder__Tensor(self, other);
}

Tensor bitwise_and(const Tensor & self, const Tensor & other) {
    return wrapper_bitwise_and_Tensor(self, other);
}
Tensor & bitwise_and_out(Tensor & out, const Tensor & self, const Tensor & other) {
    return wrapper_bitwise_and_out_out(self, other, out);
}
Tensor & bitwise_and_(Tensor & self, const Tensor & other) {
    return wrapper_bitwise_and__Tensor(self, other);
}

Tensor bitwise_or(const Tensor & self, const Tensor & other) {
    return wrapper_bitwise_or_Tensor(self, other);
}
Tensor & bitwise_or_out(Tensor & out, const Tensor & self, const Tensor & other) {
    return wrapper_bitwise_or_out_out(self, other, out);
}
Tensor & bitwise_or_(Tensor & self, const Tensor & other) {
    return wrapper_bitwise_or__Tensor(self, other);
}

Tensor bitwise_xor(const Tensor & self, const Tensor & other) {
    return wrapper_bitwise_xor_Tensor(self, other);
}
Tensor & bitwise_xor_out(Tensor & out, const Tensor & self, const Tensor & other) {
    return wrapper_bitwise_xor_out_out(self, other, out);
}
Tensor & bitwise_xor_(Tensor & self, const Tensor & other) {
    return wrapper_bitwise_xor__Tensor(self, other);
}

Tensor neg(const Tensor & self) {
    return wrapper_neg(self);
}
Tensor & neg_out(Tensor & out, const Tensor & self) {
    return wrapper_neg_out(self, out);
}
Tensor & neg_(Tensor & self) {
    return wrapper_neg_(self);
}

Tensor bitwise_not(const Tensor & self) {
    return wrapper_bitwise_not(self);
}
Tensor & bitwise_not_out(Tensor & out, const Tensor & self) {
    return wrapper_bitwise_not_out(self, out);
}
Tensor & bitwise_not_(Tensor & self) {
    return wrapper_bitwise_not_(self);
}

Tensor abs(const Tensor & self) {
    return wrapper_abs(self);
}
Tensor & abs_out(Tensor & out, const Tensor & self) {
    return wrapper_abs_out(self, out);
}
Tensor & abs_(Tensor & self) {
    return wrapper_abs_(self);
}

Tensor sin(const Tensor & self) {
    return wrapper_sin(self);
}
Tensor & sin_out(Tensor & out, const Tensor & self) {
    return wrapper_sin_out(self, out);
}
Tensor & sin_(Tensor & self) {
    return wrapper_sin_(self);
}

Tensor cos(const Tensor & self) {
    return wrapper_cos(self);
}
Tensor & cos_out(Tensor & out, const Tensor & self) {
    return wrapper_cos_out(self, out);
}
Tensor & cos_(Tensor & self) {
    return wrapper_cos_(self);
}

Tensor tan(const Tensor & self) {
    return wrapper_tan(self);
}
Tensor & tan_out(Tensor & out, const Tensor & self) {
    return wrapper_tan_out(self, out);
}
Tensor & tan_(Tensor & self) {
    return wrapper_tan_(self);
}

Tensor exp(const Tensor & self) {
    return wrapper_exp(self);
}
Tensor & exp_out(Tensor & out, const Tensor & self) {
    return wrapper_exp_out(self, out);
}
Tensor & exp_(Tensor & self) {
    return wrapper_exp_(self);
}

Tensor sqrt(const Tensor & self) {
    return wrapper_sqrt(self);
}
Tensor & sqrt_out(Tensor & out, const Tensor & self) {
    return wrapper_sqrt_out(self, out);
}
Tensor & sqrt_(Tensor & self) {
    return wrapper_sqrt_(self);
}

Tensor sigmoid(const Tensor & self) {
    return wrapper_sigmoid(self);
}
Tensor & sigmoid_out(Tensor & out, const Tensor & self) {
    return wrapper_sigmoid_out(self, out);
}
Tensor & sigmoid_(Tensor & self) {
    return wrapper_sigmoid_(self);
}

Tensor addmm(const Tensor & self, const Tensor & mat1, const Tensor & mat2, const Scalar & beta, const Scalar & alpha) {
    return wrapper_addmm(self, mat1, mat2, beta, alpha);
}
Tensor & addmm_out(Tensor & out, const Tensor & self, const Tensor & mat1, const Tensor & mat2, const Scalar & beta, const Scalar & alpha) {
    return wrapper_addmm_out(self, mat1, mat2, beta, alpha, out);
}
Tensor & addmm_(Tensor & self, const Tensor & mat1, const Tensor & mat2, const Scalar & beta, const Scalar & alpha) {
    return wrapper_addmm_(self, mat1, mat2, beta, alpha);
}

Tensor mm(const Tensor & self, const Tensor & other) {
    return wrapper_mm(self, other);
}
Tensor & mm_out(Tensor & out, const Tensor & self, const Tensor & other) {
    return wrapper_mm_out(self, other, out);
}
Tensor & mm_(Tensor & self, const Tensor & other) {
    return wrapper_mm_(self, other);
}

Tensor leaky_relu(const Tensor & self, const Scalar& negative_slope) {
    return wrapper_leaky_relu(self, negative_slope);
}
Tensor & leaky_relu_out(Tensor & out, Tensor & self, const Scalar & negative_slope) {
    return wrapper_leaky_relu_out(self, negative_slope, out);
}
Tensor & leaky_relu_(Tensor & self, const Scalar & negative_slope) {
    return wrapper_leaky_relu_(self, negative_slope);
}

std::tuple<Tensor, Tensor> max_pool2d_with_indices(const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode) {
    return wrapper_max_pool2d_with_indices(self, kernel_size, stride, padding, dilation, ceil_mode);
}

std::tuple<Tensor &, Tensor &> max_pool2d_with_indices_out(Tensor & out, Tensor & indices, const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode) {
    return wrapper_max_pool2d_with_indices_out_out(self, kernel_size, stride, padding, dilation, ceil_mode, out, indices);
}

Tensor upsample_nearest2d(const Tensor & self, IntArrayRef output_size, double scales_h, double scales_w) {
    return wrapper_upsample_nearest2d(self, output_size, scales_h, scales_w);
}

Tensor & upsample_nearest2d_out(Tensor & out, const Tensor & self, IntArrayRef output_size, double scales_h, double scales_w) {
    return wrapper_upsample_nearest2d_out_out(self, output_size, scales_h, scales_w, out);
}

Tensor upsample_bilinear2d(const Tensor & self, IntArrayRef output_size, bool align_corners, double scales_h, double scales_w) {
    return wrapper_upsample_bilinear2d(self, output_size, align_corners, scales_h, scales_w);
}

Tensor & upsample_bilinear2d_out(Tensor & out, const Tensor & self, IntArrayRef output_size, bool align_corners, double scales_h, double scales_w) {
    return wrapper_upsample_bilinear2d_out_out(self, output_size, align_corners, scales_h, scales_w, out);
}

Tensor threshold(const Tensor & self, const Scalar & threshold, const Scalar & value) {
    return wrapper_threshold(self, threshold, value);
}

Tensor & threshold_out(Tensor & out, const Tensor & self, const Scalar & threshold, const Scalar & value) {
    return wrapper_threshold_out_out(self, threshold, value, out);
}

Tensor & threshold_(Tensor & self, const Scalar & threshold, const Scalar & value) {
    return wrapper_threshold_(self, threshold, value);
}

Tensor eq(const Tensor & self, const Scalar & other) {
    return wrapper_eq_Scalar(self, other);
}
Tensor & eq_out(Tensor & out, const Tensor & self, const Scalar & other) {
    return wrapper_eq_out_Scalar_out(self, other, out);
}
Tensor & eq_outf(const Tensor & self, const Scalar & other, Tensor & out) {
    return wrapper_eq_out_Scalar_out(self, other, out);
}
Tensor & eq_(Tensor & self, const Scalar & other) {
    return wrapper_eq__Scalar(self, other);
}
Tensor eq(const Tensor & self, const Tensor & other) {
    return wrapper_eq_Tensor(self, other);
}
Tensor & eq_out(Tensor & out, const Tensor & self, const Tensor & other) {
    return wrapper_eq_out_Tensor_out(self, other, out);
}
Tensor & eq_outf(const Tensor & self, const Tensor & other, Tensor & out) {
    return wrapper_eq_out_Tensor_out(self, other, out);
}
Tensor & eq_(Tensor & self, const Tensor & other) {
    return wrapper_eq__Tensor(self, other);
}

Tensor ne(const Tensor & self, const Scalar & other) {
    return wrapper_ne_Scalar(self, other);
}
Tensor & ne_out(Tensor & out, const Tensor & self, const Scalar & other) {
    return wrapper_ne_out_Scalar_out(self, other, out);
}
Tensor & ne_outf(const Tensor & self, const Scalar & other, Tensor & out) {
    return wrapper_ne_out_Scalar_out(self, other, out);
}
Tensor & ne_(Tensor & self, const Scalar & other) {
    return wrapper_ne__Scalar(self, other);
}
Tensor ne(const Tensor & self, const Tensor & other) {
    return wrapper_ne_Tensor(self, other);
}
Tensor & ne_out(Tensor & out, const Tensor & self, const Tensor & other) {
    return wrapper_ne_out_Tensor_out(self, other, out);
}
Tensor & ne_outf(const Tensor & self, const Tensor & other, Tensor & out) {
    return wrapper_ne_out_Tensor_out(self, other, out);
}
Tensor & ne_(Tensor & self, const Tensor & other) {
    return wrapper_ne__Tensor(self, other);
}
Tensor ge(const Tensor & self, const Scalar & other) {
    return wrapper_ge_Scalar(self, other);
}
Tensor & ge_out(Tensor & out, const Tensor & self, const Scalar & other) {
    return wrapper_ge_out_Scalar_out(self, other, out);
}
Tensor & ge_outf(const Tensor & self, const Scalar & other, Tensor & out) {
    return wrapper_ge_out_Scalar_out(self, other, out);
}
Tensor & ge_(Tensor & self, const Scalar & other) {
    return wrapper_ge__Scalar(self, other);
}
Tensor ge(const Tensor & self, const Tensor & other) {
    return wrapper_ge_Tensor(self, other);
}
Tensor & ge_out(Tensor & out, const Tensor & self, const Tensor & other) {
    return wrapper_ge_out_Tensor_out(self, other, out);
}
Tensor & ge_outf(const Tensor & self, const Tensor & other, Tensor & out) {
    return wrapper_ge_out_Tensor_out(self, other, out);
}
Tensor & ge_(Tensor & self, const Tensor & other) {
    return wrapper_ge__Tensor(self, other);
}
Tensor le(const Tensor & self, const Scalar & other) {
    return wrapper_le_Scalar(self, other);
}
Tensor & le_out(Tensor & out, const Tensor & self, const Scalar & other) {
    return wrapper_le_out_Scalar_out(self, other, out);
}
Tensor & le_outf(const Tensor & self, const Scalar & other, Tensor & out) {
    return wrapper_le_out_Scalar_out(self, other, out);
}
Tensor & le_(Tensor & self, const Scalar & other) {
    return wrapper_le__Scalar(self, other);
}
Tensor le(const Tensor & self, const Tensor & other) {
    return wrapper_le_Tensor(self, other);
}
Tensor & le_out(Tensor & out, const Tensor & self, const Tensor & other) {
    return wrapper_le_out_Tensor_out(self, other, out);
}
Tensor & le_outf(const Tensor & self, const Tensor & other, Tensor & out) {
    return wrapper_le_out_Tensor_out(self, other, out);
}
Tensor & le_(Tensor & self, const Tensor & other) {
    return wrapper_le__Tensor(self, other);
}
Tensor gt(const Tensor & self, const Scalar & other) {
    return wrapper_gt_Scalar(self, other);
}
Tensor & gt_out(Tensor & out, const Tensor & self, const Scalar & other) {
    return wrapper_gt_out_Scalar_out(self, other, out);
}
Tensor & gt_outf(const Tensor & self, const Scalar & other, Tensor & out) {
    return wrapper_gt_out_Scalar_out(self, other, out);
}
Tensor & gt_(Tensor & self, const Scalar & other) {
    return wrapper_gt__Scalar(self, other);
}
Tensor gt(const Tensor & self, const Tensor & other) {
    return wrapper_gt_Tensor(self, other);
}
Tensor & gt_out(Tensor & out, const Tensor & self, const Tensor & other) {
    return wrapper_gt_out_Tensor_out(self, other, out);
}
Tensor & gt_outf(const Tensor & self, const Tensor & other, Tensor & out) {
    return wrapper_gt_out_Tensor_out(self, other, out);
}
Tensor & gt_(Tensor & self, const Tensor & other) {
    return wrapper_gt__Tensor(self, other);
}
Tensor lt(const Tensor & self, const Scalar & other) {
    return wrapper_lt_Scalar(self, other);
}
Tensor & lt_out(Tensor & out, const Tensor & self, const Scalar & other) {
    return wrapper_lt_out_Scalar_out(self, other, out);
}
Tensor & lt_outf(const Tensor & self, const Scalar & other, Tensor & out) {
    return wrapper_lt_out_Scalar_out(self, other, out);
}
Tensor & lt_(Tensor & self, const Scalar & other) {
    return wrapper_lt__Scalar(self, other);
}
Tensor lt(const Tensor & self, const Tensor & other) {
    return wrapper_lt_Tensor(self, other);
}
Tensor & lt_out(Tensor & out, const Tensor & self, const Tensor & other) {
    return wrapper_lt_out_Tensor_out(self, other, out);
}
Tensor & lt_outf(const Tensor & self, const Tensor & other, Tensor & out) {
    return wrapper_lt_out_Tensor_out(self, other, out);
}
Tensor & lt_(Tensor & self, const Tensor & other) {
    return wrapper_lt__Tensor(self, other);
}

Tensor _softmax(const Tensor & self, int64_t dim, bool half_to_float) {
    return wrapper__softmax(self, dim, half_to_float);
}
Tensor & _softmax_out(Tensor & out, const Tensor & self, int64_t dim, bool half_to_float) {
    return wrapper__softmax_out_out(self, dim, half_to_float, out);
}
Tensor & _softmax_outf(const Tensor & self, int64_t dim, bool half_to_float, Tensor & out) {
    return wrapper__softmax_out_out(self, dim, half_to_float, out);
}

::std::tuple<Tensor, Tensor> sort(const Tensor & self, bool stable, int64_t dim, bool descending) {
    return wrapper_sort_stable(self, stable, dim, descending);
}
::std::tuple<Tensor &, Tensor &> sort_out(Tensor & values, Tensor & indices, const Tensor & self, bool stable, int64_t dim, bool descending) {
    return wrapper_sort_out_values_stable(self, stable, dim, descending, values, indices);
}
::std::tuple<Tensor &,Tensor &> sort_outf(const Tensor & self, bool stable, int64_t dim, bool descending, Tensor & values, Tensor & indices) {
    return wrapper_sort_out_values_stable(self, stable, dim, descending, values, indices);
}

::std::tuple<Tensor,Tensor> topk(const Tensor & self, int64_t k, int64_t dim, bool largest, bool sorted) {
    return wrapper_topk(self, k, dim, largest, sorted);
}
::std::tuple<Tensor &,Tensor &> topk_out(Tensor & values, Tensor & indices, const Tensor & self, int64_t k, int64_t dim, bool largest, bool sorted) {
    return wrapper_topk_out_values(self, k, dim, largest, sorted, values, indices);
}
::std::tuple<Tensor &,Tensor &> topk_outf(const Tensor & self, int64_t k, int64_t dim, bool largest, bool sorted, Tensor & values, Tensor & indices) {
    return wrapper_topk_out_values(self, k, dim, largest, sorted, values, indices);
}

Tensor scatter(const Tensor & self, int64_t dim, const Tensor & index, const Tensor & src) {
    return wrapper_scatter_src(self, dim, index, src);
}
Tensor & scatter_out(Tensor & out, const Tensor & self, int64_t dim, const Tensor & index, const Tensor & src) {
    return wrapper_scatter_out_src_out(self, dim, index, src, out);
}
Tensor & scatter_outf(const Tensor & self, int64_t dim, const Tensor & index, const Tensor & src, Tensor & out) {
    return wrapper_scatter_out_src_out(self, dim, index, src, out);
}
Tensor & scatter_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & src) {
    return wrapper_scatter__src(self, dim, index, src);
}
Tensor scatter(const Tensor & self, int64_t dim, const Tensor & index, const Scalar & value) {
    return wrapper_scatter_value(self, dim, index, value);
}
Tensor & scatter_out(Tensor & out, const Tensor & self, int64_t dim, const Tensor & index, const Scalar & value) {
    return wrapper_scatter_out_value_out(self, dim, index, value, out);
}
Tensor & scatter_outf(const Tensor & self, int64_t dim, const Tensor & index, const Scalar & value, Tensor & out) {
    return wrapper_scatter_out_value_out(self, dim, index, value, out);
}
Tensor & scatter_(Tensor & self, int64_t dim, const Tensor & index, const Scalar & value) {
    return wrapper_scatter__value(self, dim, index, value);
}
Tensor scatter(const Tensor & self, int64_t dim, const Tensor & index, const Tensor & src, int64_t reduce) {
    return wrapper_scatter_reduce(self, dim, index, src, reduce);
}
Tensor & scatter_out(Tensor & out, const Tensor & self, int64_t dim, const Tensor & index, const Tensor & src, int64_t reduce) {
    return wrapper_scatter_out_reduce_out(self, dim, index, src, reduce, out);
}
Tensor & scatter_outf(const Tensor & self, int64_t dim, const Tensor & index, const Tensor & src, int64_t reduce, Tensor & out) {
    return wrapper_scatter_out_reduce_out(self, dim, index, src, reduce, out);
}
Tensor & scatter_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & src, int64_t reduce) {
    return wrapper_scatter__reduce(self, dim, index, src, reduce);
}
Tensor scatter(const Tensor & self, int64_t dim, const Tensor & index, const Scalar & value, int64_t reduce) {
    return wrapper_scatter_value_reduce(self, dim, index, value, reduce);
}
Tensor & scatter_out(Tensor & out, const Tensor & self, int64_t dim, const Tensor & index, const Scalar & value, int64_t reduce) {
    return wrapper_scatter_out_value_reduce_out(self, dim, index, value, reduce, out);
}
Tensor & scatter_outf(const Tensor & self, int64_t dim, const Tensor & index, const Scalar & value, int64_t reduce, Tensor & out) {
    return wrapper_scatter_out_value_reduce_out(self, dim, index, value, reduce, out);
}
Tensor & scatter_(Tensor & self, int64_t dim, const Tensor & index, const Scalar & value, int64_t reduce) {
    return wrapper_scatter__value_reduce(self, dim, index, value, reduce);
}

Tensor baddbmm(const Tensor & self, const Tensor & batch1, const Tensor & batch2, const Scalar & beta, const Scalar & alpha) {
    return wrapper_baddbmm(self, batch1, batch2, beta, alpha);
}
Tensor & baddbmm_out(Tensor & out, const Tensor & self, const Tensor & batch1, const Tensor & batch2, const Scalar & beta, const Scalar & alpha) {
    return wrapper_baddbmm_out_out(self, batch1, batch2, beta, alpha, out);
}
Tensor & baddbmm_outf(const Tensor & self, const Tensor & batch1, const Tensor & batch2, const Scalar & beta, const Scalar & alpha, Tensor & out) {
    return wrapper_baddbmm_out_out(self, batch1, batch2, beta, alpha, out);
}
Tensor & baddbmm_(Tensor & self, const Tensor & batch1, const Tensor & batch2, const Scalar & beta, const Scalar & alpha) {
    return wrapper_baddbmm_(self, batch1, batch2, beta, alpha);
}

Tensor bmm(const Tensor & self, const Tensor & mat2) {
    return wrapper_bmm(self, mat2);
}
Tensor & bmm_out(Tensor & out, const Tensor & self, const Tensor & mat2) {
    return wrapper_bmm_out_out(self, mat2, out);
}
Tensor & bmm_outf(const Tensor & self, const Tensor & mat2, Tensor & out) {
    return wrapper_bmm_out_out(self, mat2, out);
}

Tensor sum(const Tensor & self, IntArrayRef dim, bool keepdim, ScalarType dtype) {
    return wrapper_sum_dim_IntList(self, dim, keepdim, dtype);
}
Tensor & sum_out(Tensor & out, const Tensor & self, IntArrayRef dim, bool keepdim, ScalarType dtype) {
    return wrapper_sum_out_IntList_out(self, dim, keepdim, dtype, out);
}
Tensor & sum_outf(const Tensor & self, IntArrayRef dim, bool keepdim, ScalarType dtype, Tensor & out) {
    return wrapper_sum_out_IntList_out(self, dim, keepdim, dtype, out);
}

Tensor index(const Tensor & self, const std::vector<otter::optional<Tensor>> & indices) {
    return wrapper_index_Tensor(self, indices);
}
Tensor & index_out(Tensor & out, const Tensor & self, const std::vector<otter::optional<Tensor>> & indices) {
    return wrapper_index_out_Tensor_out(self, indices, out);
}
Tensor & index_outf(const Tensor & self, const std::vector<otter::optional<Tensor>> & indices, Tensor & out) {
    return wrapper_index_out_Tensor_out(self, indices, out);
}

Tensor prod(const Tensor & self, int64_t dim, bool keepdim, ScalarType dtype) {
    return wrapper_prod_dim_int(self, dim, keepdim, dtype);
}
Tensor & prod_out(Tensor & out, const Tensor & self, int64_t dim, bool keepdim, ScalarType dtype) {
    return wrapper_prod_out_int_out(self, dim, keepdim, dtype, out);
}
Tensor & prod_outf(const Tensor & self, int64_t dim, bool keepdim, ScalarType dtype, Tensor & out) {
    return wrapper_prod_out_int_out(self, dim, keepdim, dtype, out);
}

Tensor mean(const Tensor & self, IntArrayRef dim, bool keepdim, ScalarType dtype) {
    return wrapper_mean_dim(self, dim, keepdim, dtype);
}
Tensor & mean_out(Tensor & out, const Tensor & self, IntArrayRef dim, bool keepdim, ScalarType dtype) {
    return wrapper_mean_out_out(self, dim, keepdim, dtype, out);
}
Tensor & mean_outf(const Tensor & self, IntArrayRef dim, bool keepdim, ScalarType dtype, Tensor & out) {
    return wrapper_mean_out_out(self, dim, keepdim, dtype, out);
}

}   // end namespace native
}   // end namesapce otter
