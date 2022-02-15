//
//  TensorLinearAlgebra.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/2/11.
//

#include "Dispatch.hpp"
#include "TensorResize.hpp"
#include "TensorBlas.hpp"
#include "TensorLinearAlgebra.hpp"
#include "Formatting.hpp"
#include "TensorFunction.hpp"
#include "ExpandUtils.hpp"

namespace otter {

DEFINE_META_FUNCTION(addmm) (const Tensor& self, const Tensor& mat1, const Tensor& mat2, const Scalar& beta, const Scalar& alpha) {
    // Ensure that it is 2D matrix
    assert(mat1.dim() == 2);
    assert(mat2.dim() == 2);
    // Check the shape of mat1 and mat2 can be multiplied
    assert(mat1.sizes()[1] == mat2.sizes()[0]);
    set_output(0, {mat1.sizes()[0], mat2.sizes()[1]}, {}, self.options());
}

DEFINE_IMPL_FUNCTION(addmm_out_cpu) (const Tensor & self, const Tensor & mat1, const Tensor & mat2, const Scalar & beta, const Scalar & alpha, const Tensor & out) {
    auto b_self = expand_size(self, {mat1.sizes()[0], mat2.sizes()[1]});
    addmm_impl_cpu_(const_cast<Tensor&>(out), *b_self, mat1, mat2, beta, alpha);
}

DEFINE_META_FUNCTION(mm) (const Tensor& self, const Tensor& other) {
    // Ensure that it is 2D matrix
    assert(self.dim() == 2);
    assert(other.dim() == 2);
    // Check the shape of mat1 and mat2 can be multiplied
    assert(self.sizes()[1] == other.sizes()[0]);
    set_output(0, {self.sizes()[0], other.sizes()[1]}, {}, self.options());
}

DEFINE_IMPL_FUNCTION(mm_out_cpu) (const Tensor& self, const Tensor& other, const Tensor& out) {
    addmm_impl_cpu_(const_cast<Tensor&>(out), out, self, other, 0, 1);
}

void addmm_impl_cpu_(Tensor &result, const Tensor &self, Tensor m1, Tensor m2, const Scalar& beta, const Scalar& alpha) {
    // Ensure that is a 2D matrix
    assert(self.dim() == 2 && m1.dim() == 2 && m2.dim() == 2);
    const auto self_sizes = self.sizes();
    auto m1_strides = m1.strides();
    auto m1_sizes = m1.sizes();
    auto m2_strides = m2.strides();
    auto m2_sizes = m2.sizes();
    // Ensure the matrix can do matrix multiplication
    assert(self_sizes[0] == m1_sizes[0] && self_sizes[1] == m2_sizes[1]);
    
    otter::resize_output(result, self_sizes);
    const auto result_strides = result.strides();
    const auto result_sizes = result.sizes();

    if (result.numel() == 0) {
        return;
    }
    
//    if (beta.toComplexDouble() != 0.0 && !self.is_same(result)) {
//        result.copy_(self);
//    }
    
    bool transpose_c = false;
    Tensor c;
    
    if (result_strides[0] == 1 && (result_sizes[1] == 1 || result_strides[1] >= std::max(int64_t{1}, result_sizes[0]))) {
        transpose_c = false;
        c = result; // resolve_conj()
    } else if (result_strides[1] == 1 && (result_sizes[0] == 1 || result_strides[0] >= std::max(int64_t{1}, result_sizes[1]))) {
        std::swap(m1, m2);
        std::swap(m1_sizes, m2_sizes);
        std::swap(m1_strides, m2_strides);
        transpose_c = true;
        c = result; // resolve_conj()
    } else {
        transpose_c = false;
        // make c FORTRAN contiguous
        c = result.transpose(0, 1).contiguous().transpose_(0, 1);
    }
    
    const int64_t m = result_sizes[transpose_c ? 1 : 0];
    const int64_t n = result_sizes[transpose_c ? 0 : 1];
    const int64_t k = m1_sizes[transpose_c ? 0 : 1];
    
    bool transpose_a = false;
    Tensor a;
    
    if (m1_strides[transpose_c ? 1 : 0] == 1 && m1_strides[transpose_c ? 0 : 1] >= std::max(int64_t{1}, m)) {
        transpose_a = false;
        a = m1; // resolve_conj()
    } else if (m1_strides[transpose_c ? 0 : 1] == 1 && m1_strides[transpose_c ? 1 : 0] >= std::max(int64_t{1}, k)) {
        transpose_a = true;
        a = m1;
    } else {
        transpose_a = !transpose_c;
        a = m1.clone(MemoryFormat::Contiguous);
    }
    
    bool transpose_b = false;
    Tensor b;
    
    if (m2_strides[transpose_c ? 1 : 0] == 1 && m2_strides[transpose_c ? 0 : 1] >= std::max(int64_t{1}, k)) {
        transpose_b = false;
        b = m2; // resolve_conj()
    } else if (m2_strides[transpose_c ? 0 : 1] == 1 && m2_strides[transpose_c ? 1 : 0] >= std::max(int64_t{1}, n)) {
        transpose_b = true;
        b = m2;
    } else {
        transpose_b = !transpose_c;
        b = m2.clone(MemoryFormat::Contiguous);
    }
    
    const int64_t lda = a.strides()[(transpose_a == transpose_c) ? 1 : 0];
    const int64_t ldb = b.strides()[(transpose_b == transpose_c) ? 1 : 0];
    const int64_t ldc = c.strides()[transpose_c ? 0 : 1];
    
    OTTER_DISPATCH_ALL_TYPES(result.scalar_type(), "addmm_impl_cpu_", [&] {
        otter::gemm(
            transpose_a ? TransposeType::Transpose : TransposeType::NoTranspose,
            transpose_b ? TransposeType::Transpose : TransposeType::NoTranspose,
            m, n, k,
            alpha.to<scalar_t>(),
            a.data_ptr<scalar_t>(), lda,
            b.data_ptr<scalar_t>(), ldb,
            beta.to<scalar_t>(),
            c.data_ptr<scalar_t>(), ldc);
    });
    if (!c.is_same(result)) {
        result.copy_(c);
    }
}


}
