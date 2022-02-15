//
//  TensorBlas.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/2/11.
//

#include "TensorBlas.hpp"
#include "Dispatch.hpp"
#include "EmptyTensor.hpp"
#include "TensorBlasKernel.hpp"

namespace otter {

DEFINE_DISPATCH(gemm_stub);

#define INSTANTIATE_GEMM(T, S)                                          \
template <>                                                             \
void gemm(                                                              \
    TransposeType transa, TransposeType transb,                         \
    int64_t m, int64_t n, int64_t k,                                    \
    const T alpha,                                                      \
    const T *a, int64_t lda,                                            \
    const T *b, int64_t ldb,                                            \
    const T beta,                                                       \
    T *c, int64_t ldc) {                                                \
    normalize_last_dims(transa, transb, m, n, k, &lda, &ldb, &ldc);     \
    gemm_stub(                                                          \
        Device::CPU, ScalarType::S,                                     \
        transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);  \
}
OTTER_ALL_SCALAR_TYPES(INSTANTIATE_GEMM)
#undef INSTANTIATE_GEMM

void normalize_last_dims(
    TransposeType transa, TransposeType transb,
    int64_t m, int64_t n, int64_t k,
    int64_t *lda, int64_t *ldb, int64_t *ldc) {
    if (n == 1) {
        *ldc = m;
    }

    if(transa != TransposeType::NoTranspose) {
        if (m == 1) {
            *lda = k;
        }
    } else if(k == 1) {
        *lda = m;
    }

    if(transb != TransposeType::NoTranspose) {
        if (k == 1) {
            *ldb = n;
        }
    } else if (n == 1) {
        *ldb = k;
    }
}

inline void dot_check(const Tensor& self, const Tensor& other) {
    // 1D tensor expected
    assert(self.dim() == 1 && other.dim() == 1);
    // Same dtype expected
    assert(self.dtype() == other.dtype());
    // Same size expected
    assert(self.numel() == other.numel());
}

Tensor dot(const Tensor& self, const Tensor& other) {
    dot_check(self, other);
    
    return OTTER_DISPATCH_ALL_TYPES(self.scalar_type(), "dot", [&] {
        Tensor result = otter::empty_cpu({}, self.options());
        result.fill_(otter::dot_impl(self.numel(), self.data_ptr<scalar_t>(), self.stride(0), other.data_ptr<scalar_t>(), other.stride(0)));
        return result;
    });
}



}
