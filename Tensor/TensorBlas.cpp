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
#include "TensorResize.hpp"

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

template <typename scalar_t>
void gemm_batched_generic(
    TransposeType transa, TransposeType transb,
    int64_t batch_size, int64_t m, int64_t n, int64_t k,
    scalar_t alpha,
    const scalar_t **a, int64_t lda,
    const scalar_t **b, int64_t ldb,
    scalar_t beta,
    scalar_t **c, int64_t ldc) {
    for (const auto batch : otter::irange(batch_size)) {
        gemm(transa, transb, m, n, k, alpha, a[batch], lda, b[batch], ldb, beta, c[batch], ldc);
    }
}
template <typename scalar_t>
void gemm_batched(
    TransposeType transa, TransposeType transb,
    int64_t batch_size, int64_t m, int64_t n, int64_t k,
    scalar_t alpha,
    const scalar_t **a, int64_t lda,
    const scalar_t **b, int64_t ldb,
    scalar_t beta,
    scalar_t **c, int64_t ldc) {
    if (batch_size == 1) {
        return gemm(transa, transb, m, n, k, alpha, a[0], lda, b[0], ldb, beta, c[0], ldc);
    }
    gemm_batched_generic(transa, transb, batch_size, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}
template <typename scalar_t>
void gemm_batched_with_stride_generic(
    TransposeType transa, TransposeType transb,
    int64_t batch_size, int64_t m, int64_t n, int64_t k,
    scalar_t alpha,
    const scalar_t *a, int64_t lda, int64_t batch_stride_a,
    const scalar_t *b, int64_t ldb, int64_t batch_stride_b,
    scalar_t beta,
    scalar_t *c, int64_t ldc, int64_t batch_stride_c) {
    for (const auto batch : otter::irange(batch_size)) {
        const auto a_batch = a + batch_stride_a * batch;
        const auto b_batch = b + batch_stride_b * batch;
        const auto c_batch = c + batch_stride_c * batch;
        gemm(transa, transb, m, n, k, alpha, a_batch, lda, b_batch, ldb, beta, c_batch, ldc);
    }
}
template <typename scalar_t>
void gemm_batched_with_stride(
    TransposeType transa, TransposeType transb,
    int64_t batch_size, int64_t m, int64_t n, int64_t k,
    scalar_t alpha,
    const scalar_t *a, int64_t lda, int64_t batch_stride_a,
    const scalar_t *b, int64_t ldb, int64_t batch_stride_b,
    scalar_t beta,
    scalar_t *c, int64_t ldc, int64_t batch_stride_c) {
    if (batch_size == 1) {
        return gemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    }
    gemm_batched_with_stride_generic(
        transa, transb, batch_size, m, n, k, alpha, a, lda, batch_stride_a,
        b, ldb, batch_stride_b, beta, c, ldc, batch_stride_c);
}
#define INSTANTIATE_BATCHED_GEMM(scalar_t, DType)               \
  template void gemm_batched(                                   \
      TransposeType transa, TransposeType transb,               \
      int64_t batch_size, int64_t m, int64_t n, int64_t k,      \
      scalar_t alpha,                                           \
      const scalar_t **a, int64_t lda,                          \
      const scalar_t **b, int64_t ldb,                          \
      scalar_t beta,                                            \
      scalar_t **c, int64_t ldc);                               \
  template void gemm_batched_with_stride(                       \
      TransposeType transa, TransposeType transb,               \
      int64_t batch_size, int64_t m, int64_t n, int64_t k,      \
      scalar_t alpha,                                           \
      const scalar_t *a, int64_t lda, int64_t batch_stride_a,   \
      const scalar_t *b, int64_t ldb, int64_t batch_stride_b,   \
      scalar_t beta,                                            \
      scalar_t *c, int64_t ldc, int64_t batch_stride_c);
OTTER_ALL_SCALAR_TYPES(INSTANTIATE_BATCHED_GEMM)

inline void dot_check(const Tensor& self, const Tensor& other) {
    OTTER_CHECK(self.dim() == 1 && other.dim() == 1, "1D tensor expected");
    OTTER_CHECK(self.dtype() == other.dtype(), "Same dtype expected");
    OTTER_CHECK(self.numel() == other.numel(), "Same size expected");
}

Tensor dot(const Tensor& self, const Tensor& other) {
    dot_check(self, other);
    
    return OTTER_DISPATCH_ALL_TYPES(self.scalar_type(), "dot", [&] {
        Tensor result = otter::empty_cpu({}, self.options());
        result.fill_(otter::dot_impl(self.numel(), self.data_ptr<scalar_t>(), self.stride(0), other.data_ptr<scalar_t>(), other.stride(0)));
        return result;
    });
}

Tensor& dot_out(const Tensor& self, const Tensor& other, Tensor& result) {
    auto output_device = result.device();
    auto input1_device = self.device();
    auto input2_device = other.device();
    // check if the input & output tensors are on the same device.
    OTTER_CHECK((output_device == input1_device) && (input1_device == input2_device),
                "dot: Expected the output and input tensors to be on the "
                "same device");
    otter::native::resize_output(result, {});
    OTTER_CHECK(result.scalar_type() == self.scalar_type(),
                "result dtype does not match input dtype ");
    return result.fill_(self.dot(other));
}



}
