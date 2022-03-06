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
    assert(out.dtype() == options.dtype());
    assert(out.device() == options.device());
    
    const bool resized = resize_output(out, sizes);
    if (resized) {
        if (!strides.empty()) {
            
        }
    }
}

void check_inplace(const Tensor &self, IntArrayRef sizes, const TensorOptions &options) {
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
    void set_output(int64_t output_idx, IntArrayRef sizes, IntArrayRef strides, TensorOptions options) override {
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
    void set_output(int64_t output_idx, IntArrayRef sizes, IntArrayRef strides, TensorOptions options) override {
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

// upsample
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

// end upsample

namespace cpu {

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

}   // end namespace cpu
}   // end namesapce otter
