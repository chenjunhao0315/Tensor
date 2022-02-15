//
//  TensorBlas.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/2/11.
//

#ifndef TensorBlas_hpp
#define TensorBlas_hpp

#include "DispatchStub.hpp"
#include "Tensor.hpp"

namespace otter {

Tensor dot(const Tensor& self, const Tensor& other);

enum class TransposeType {
    NoTranspose,
    Transpose,
    ConjTranspose,
};

void normalize_last_dims(
  TransposeType transa, TransposeType transb,
  int64_t m, int64_t n, int64_t k,
  int64_t *lda, int64_t *ldb, int64_t *ldc);

using gemm_fn = void(*)(
    ScalarType type,
    TransposeType transa, TransposeType transb,
    int64_t m, int64_t n, int64_t k,
    const Scalar& alpha,
    const void *a, int64_t lda,
    const void *b, int64_t ldb,
    const Scalar& beta,
    void *c, int64_t ldc);

DECLARE_DISPATCH(gemm_fn, gemm_stub);

template <typename scalar_t>
void gemm(
    TransposeType transa, TransposeType transb,
    int64_t m, int64_t n, int64_t k,
    const scalar_t alpha,
    const scalar_t *a, int64_t lda,
    const scalar_t *b, int64_t ldb,
    const scalar_t beta,
    scalar_t *c, int64_t ldc);

}

#endif /* TensorBlas_hpp */
