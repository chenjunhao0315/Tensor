//
//  TensorBlasKernel.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/2/12.
//

#include "Dispatch.hpp"
#include "TensorBlas.hpp"
#include "TensorBlasKernel.hpp"

namespace otter {

template <typename scalar_t>
void scale_(int64_t m, int64_t n, scalar_t alpha, scalar_t *a, int64_t lda) {
    if (alpha == scalar_t(1)) {
        return;  // identity
    }

    if (alpha == scalar_t(0)) {
        for (const auto j : otter::irange(n)) {
            for (const auto i : otter::irange(m)) {
                a[j * lda + i] = scalar_t(0);
            }
        }
        return;
    }

    for (const auto j : otter::irange(n)) {
        for (const auto i : otter::irange(m)) {
            a[j * lda + i] *= alpha;
        }
    }
}

template <typename scalar_t>
void gemm_nn_(
    int64_t m, int64_t n, int64_t k,
    scalar_t alpha,
    const scalar_t *a, int64_t lda,
    const scalar_t *b, int64_t ldb,
    scalar_t beta,
    scalar_t *c, int64_t ldc) {
    // c *= beta
    scale_(m, n, beta, c, ldc);

    // c += alpha * (a @ b)
    for (const auto l : otter::irange(k)) {
        for (const auto j : otter::irange(n)) {
            scalar_t val = b[l + j * ldb] * alpha;
            int64_t i_m = m / 4;
            for (const auto i_i : otter::irange(i_m)) {
                c[j * ldc + i_i * 4 + 0] += a[i_i * 4 + 0 + l * lda] * val;
                c[j * ldc + i_i * 4 + 1] += a[i_i * 4 + 1 + l * lda] * val;
                c[j * ldc + i_i * 4 + 2] += a[i_i * 4 + 2 + l * lda] * val;
                c[j * ldc + i_i * 4 + 3] += a[i_i * 4 + 3 + l * lda] * val;
            }
            int64_t i = i_m * 4;
            for (; i < m; i++)
                c[j * ldc + i] += a[i + l * lda] * val;
        }
    }
}

template <typename scalar_t>
void gemm_tn_(
    int64_t m, int64_t n, int64_t k,
    scalar_t alpha,
    const scalar_t *a, int64_t lda,
    const scalar_t *b, int64_t ldb,
    scalar_t beta,
    scalar_t *c, int64_t ldc) {
    // c = alpha * (a.T @ b) + beta * c
    const scalar_t *a_ = a;
    for (const auto i : otter::irange(m)) {
        const scalar_t *b_ = b;
        for (const auto j : otter::irange(n)) {
            scalar_t sum = 0;
            for (const auto l : otter::irange(k)) {
                sum += a_[l] * b_[l];
            }
            b_ += ldb;
            if (beta == scalar_t(0))
                c[j * ldc + i] = alpha * sum;
            else
                c[j * ldc + i] = beta * c[j * ldc + i] + alpha * sum;
        }
        a_ += lda;
    }
}

template <typename scalar_t>
void gemm_nt_(
    int64_t m, int64_t n, int64_t k,
    scalar_t alpha,
    const scalar_t *a, int64_t lda,
    const scalar_t *b, int64_t ldb,
    scalar_t beta,
    scalar_t *c, int64_t ldc) {
    // c *= beta
    scale_(m, n, beta, c, ldc);

    // c += alpha * (a @ b.T)
    for (const auto l : otter::irange(k)) {
        for (const auto j : otter::irange(n)) {
            scalar_t val = b[j + l * ldb] * alpha;
            int64_t i_m = m / 4;
            for (const auto i_i : otter::irange(i_m)) {
                c[j * ldc + i_i * 4 + 0] += a[i_i * 4 + 0 + l * lda] * val;
                c[j * ldc + i_i * 4 + 1] += a[i_i * 4 + 1 + l * lda] * val;
                c[j * ldc + i_i * 4 + 2] += a[i_i * 4 + 2 + l * lda] * val;
                c[j * ldc + i_i * 4 + 3] += a[i_i * 4 + 3 + l * lda] * val;
            }
            int64_t i = i_m * 4;
            for (; i < m; i++)
                c[j * ldc + i] += a[i + l * lda] * val;
        }
    }
}

template <typename scalar_t>
void gemm_tt_(
    int64_t m, int64_t n, int64_t k,
    scalar_t alpha,
    const scalar_t *a, int64_t lda,
    const scalar_t *b, int64_t ldb,
    scalar_t beta,
    scalar_t *c, int64_t ldc) {
    // c *= beta
    scale_(m, n, beta, c, ldc);

    // c += alpha * (a.T @ b.T)
    for (const auto i : otter::irange(m)) {
        for (const auto j : otter::irange(n)) {
            int64_t l_k = k / 4;
            for (const auto l_l : otter::irange(l_k)) {
                c[j * ldc + i] += a[i * lda + l_l * 4 + 0] //
                    * b[(l_l * 4 + 0) * ldb + j] * alpha;
                c[j * ldc + i] += a[i * lda + l_l * 4 + 1] //
                    * b[(l_l * 4 + 1) * ldb + j] * alpha;
                c[j * ldc + i] += a[i * lda + l_l * 4 + 2] //
                    * b[(l_l * 4 + 2) * ldb + j] * alpha;
                c[j * ldc + i] += a[i * lda + l_l * 4 + 3] //
                    * b[(l_l * 4 + 3) * ldb + j] * alpha;
            }
            int64_t l = l_k * 4;
            for (; l < k; l++)
                c[j * ldc + i] += a[i * lda + l] * b[l * ldb + j] * alpha;
        }
    }
}

template <typename scalar_t>
void gemm_core_(
    TransposeType transa, TransposeType transb,
    int64_t m, int64_t n, int64_t k,
    scalar_t alpha,
    const scalar_t *a, int64_t lda,
    const scalar_t *b, int64_t ldb,
    scalar_t beta,
    scalar_t *c, int64_t ldc) {
    if(transa == TransposeType::NoTranspose && transb == TransposeType::NoTranspose) {
        return gemm_nn_(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    } else if(transa == TransposeType::Transpose && transb != TransposeType::Transpose) {
        gemm_tn_(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    } else if(transa == TransposeType::NoTranspose && transb == TransposeType::Transpose) {
        gemm_nt_(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    } else {  // transa == TransposeType::Transpose && transb == TransposeType::Transpose
        gemm_tt_(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    }
}

void cpublas_gemm_impl(
    ScalarType type,
    TransposeType transa, TransposeType transb,
    int64_t m, int64_t n, int64_t k,
    const Scalar& alpha,
    const void *a, int64_t lda,
    const void *b, int64_t ldb,
    const Scalar& beta,
    void *c, int64_t ldc) {
    OTTER_DISPATCH_ALL_TYPES(type, "cpublas_gemm_impl", [&]{
        gemm_core_(
            transa, transb, m, n, k,
            alpha.to<scalar_t>(),
            static_cast<const scalar_t *>(a), lda,
            static_cast<const scalar_t *>(b), ldb,
            beta.to<scalar_t>(),
            static_cast<scalar_t *>(c), ldc);
    });
}

REGISTER_DISPATCH(gemm_stub, &cpublas_gemm_impl);


template <typename scalar_t, typename Functor>
scalar_t dot_naive(
    int64_t n,
    scalar_t* x,
    int64_t incx,
    scalar_t* y,
    int64_t incy,
    Functor op) {
    int64_t i;
    scalar_t sum = 0;
    for (i = 0; i < n; i++) {
        sum += op(x[i * incx], y[i * incy]);
    }
    return sum;
}

template <typename scalar_t>
scalar_t dot_impl_floating(int64_t n, scalar_t* x, int64_t incx, scalar_t* y, int64_t incy)
{
    if (n == 1) {
        incx = 1;
        incy = 1;
    }
    return dot_naive(n, x, incx, y, incy, std::multiplies<scalar_t>{});
}

template <typename scalar_t>
scalar_t dot_impl(int64_t n, scalar_t* x, int64_t incx, scalar_t* y, int64_t incy) {
    if (n == 1) {
        incx = 1;
        incy = 1;
    }
    return dot_naive(n, x, incx, y, incy, std::multiplies<scalar_t>{});
}

template <>
float dot_impl(int64_t n, float* x, int64_t incx, float* y, int64_t incy) {
    return otter::dot_impl_floating(n, x, incx, y, incy);
}

template <>
double dot_impl(int64_t n, double* x, int64_t incx, double* y, int64_t incy) {
    return otter::dot_impl_floating(n, x, incx, y, incy);
}

#define INSTANTIATE_DOT_IMPL(scalar_t)  \
template scalar_t dot_impl<scalar_t>( \
int64_t n, scalar_t * x, int64_t incx, scalar_t * y, int64_t incy);
INSTANTIATE_DOT_IMPL(uint8_t);
INSTANTIATE_DOT_IMPL(int8_t);
INSTANTIATE_DOT_IMPL(int16_t);
INSTANTIATE_DOT_IMPL(int);
INSTANTIATE_DOT_IMPL(int64_t);
#undef INSTANTIATE_DOT_IMPL

}   // end namespace otter
