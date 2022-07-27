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
#include "TensorFactory.hpp"
#include "AutoBuffer.hpp"
#include "TensorMaker.hpp"
#include <float.h>
#include <time.h>
#include "TensorOperator.hpp"
#include "Parallel.hpp"

namespace otter {

DEFINE_META_FUNCTION(addmm) (const Tensor& self, const Tensor& mat1, const Tensor& mat2, const Scalar& /*beta*/, const Scalar& /*alpha*/) {
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
    
    otter::native::resize_output(result, self_sizes);
    const auto result_strides = result.strides();
    const auto result_sizes = result.sizes();

    if (result.numel() == 0) {
        return;
    }
    
    if (beta.toDouble() != 0.0 && !self.is_same(result)) {
        result.copy_(self);
    }
    
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

template <typename Meta>
void common_checks_baddbmm_bmm(Meta& meta, const Tensor& batch1, const Tensor& batch2, const Scalar& /*beta*/, const Scalar& /*alpha*/, bool /*is_bmm*/, const Tensor& self_baddbmm = Tensor()) {
  OTTER_CHECK(batch1.dim() == 3, "batch1 must be a 3D tensor");
  OTTER_CHECK(batch2.dim() == 3, "batch2 must be a 3D tensor");
  const auto batch1_sizes = batch1.sizes();
  const auto batch2_sizes = batch2.sizes();
  int64_t bs = batch1_sizes[0];
  int64_t contraction_size = batch1_sizes[2];
  int64_t res_rows = batch1_sizes[1];
  int64_t res_cols = batch2_sizes[2];
  std::vector<int64_t> output_size {bs, res_rows, res_cols};
  OTTER_CHECK(batch2_sizes[0] == bs && batch2_sizes[1] == contraction_size,
              "Expected size for first two dimensions of batch2 tensor to be: [",
              bs, ", ", contraction_size, "] but got: [", batch2_sizes[0], ", ", batch2_sizes[1], "].");
//  auto& result = meta.maybe_get_output(0);
  // 'set_output' does not resize for in-place calls
  meta.set_output(0, output_size, {}, batch2.options());
//  const auto result_sizes = result.sizes();
  // Error is raised if called from in-place overload with incorrect shape
//  OTTER_CHECK(result_sizes == output_size,
//              "Expected an output tensor with shape [", output_size, "] but got shape ", result_sizes);
}
DEFINE_META_FUNCTION(bmm)(const Tensor& self, const Tensor& mat2) {
    common_checks_baddbmm_bmm(*this, self, mat2, Scalar(0.0), Scalar(1.0), true);
}
DEFINE_META_FUNCTION(baddbmm)(const Tensor& self, const Tensor& batch1, const Tensor& batch2, const Scalar& beta, const Scalar& alpha) {
  auto self_ = expand_size(self, {batch1.size(0), batch1.size(1), batch2.size(2)});
  common_checks_baddbmm_bmm(*this, batch1, batch2, beta, alpha, false, *self_);
}

static void addbmm_impl_(Tensor &result, const Tensor &self, const Tensor &batch1, const Tensor &batch2, const Scalar& beta, const Scalar& alpha) {
  OTTER_CHECK(batch1.dim() == 3, "batch1 must be a 3D tensor");
  OTTER_CHECK(batch2.dim() == 3, "batch2 must be a 3D tensor");
  OTTER_CHECK(batch1.size(0) == batch2.size(0),
      "batch1 and batch2 must have same number of batches, got ",
      batch1.size(0), " and ", batch2.size(0));
  OTTER_CHECK(batch1.size(2) == batch2.size(1),
      "Incompatible matrix sizes for bmm (",
      batch1.size(1), "x", batch1.size(2), " and ",
      batch2.size(1), "x", batch2.size(2), ")");
  const int64_t dim1 = batch1.size(1);
  const int64_t dim2 = batch2.size(2);
  OTTER_CHECK(self.size(0) == dim1 && self.size(1) == dim2,
      "self tensor does not match matmul output shape");
  result.resize_as_(self);
  if (beta.to<double>() != 0.0 && !self.is_same(result)) {
    result.copy_(self);
  }
  const int64_t num_batches = batch1.size(0);
  if (num_batches == 0) {
    if (beta.to<double>() != 0.0) {
      result.mul_(beta);
    } else {
      result.zero_();
    }
    return;
  }
  auto adjusted_beta(beta);
  for (const auto batch : otter::irange(num_batches)) {
    result.addmm_(batch1[batch], batch2[batch], adjusted_beta, alpha);
    adjusted_beta = 1; // accumulate output once
  }
}
Tensor& addbmm_out(const Tensor& self, const Tensor& batch1, const Tensor& batch2, const Scalar& beta, const Scalar& alpha, Tensor& result) {
  auto b_self = expand_size(self, {batch1.size(1), batch2.size(2)});
  {
    addbmm_impl_(result, *b_self, batch1, batch2, beta, alpha);
  }
  return result;
}
Tensor &addbmm_(Tensor& self, const Tensor& batch1, const Tensor& batch2, const Scalar& beta, const Scalar& alpha) {
  return addbmm_out(self, batch1, batch2, beta, alpha, self);
}
Tensor addbmm(const Tensor& self, const Tensor& batch1, const Tensor& batch2, const Scalar& beta, const Scalar& alpha) {
  Tensor result = otter::empty({0}, self.options());
  return addbmm_out(self, batch1, batch2, beta, alpha, result);
}

template <typename scalar_t, bool is_bmm>
inline void baddbmm_cpu_kernel(const Tensor& result, const Tensor& self, const Tensor& mat2, const Scalar& beta_, const Scalar& alpha_) {
  int64_t bs = result.size(0);
  int64_t is = result.size(1);
  int64_t js = result.size(2);
  int64_t ks = self.size(2);
  scalar_t alpha = alpha_.to<scalar_t>();
  scalar_t beta = beta_.to<scalar_t>();
  auto r0 = result.accessor<scalar_t, 3>();
  auto s0 = self.accessor<scalar_t, 3>();
  auto m0 = mat2.accessor<scalar_t, 3>();
  int64_t grain_size = std::min(otter::GRAIN_SIZE / (is * js * ks), (int64_t)1);
  parallel_for(0, bs, grain_size, [&](int64_t b_begin, int64_t b_end) {
      for (const auto b : otter::irange(b_begin, b_end)) {
        auto r1 = r0[b];
        auto s1 = s0[b];
        auto m1 = m0[b];
        for (const auto i : otter::irange(is)) {
          auto r2 = r1[i];
          auto s2 = s1[i];
          for (const auto j : otter::irange(js)) {
            scalar_t &r = r2[j];
            if (is_bmm) {
              r = 0;
              for (const auto k : otter::irange(ks)) {
                r += s2[k] * m1[k][j];
              }
            } else {
              r *= beta;
              for (const auto k : otter::irange(ks)) {
                r += alpha * s2[k] * m1[k][j];
              }
            }
          }
        }
      }
    });
}
void baddbmm_with_gemm_(const Tensor &result, const Tensor &mat1, const Tensor &mat2, const Scalar &beta_, const Scalar &alpha_) {
  OTTER_INTERNAL_ASSERT(result.is_contiguous());
  const auto result_sizes = result.sizes();
  const auto result_strides = result.strides();
  const auto mat1_strides = mat1.strides();
  const auto mat2_strides = mat2.strides();
  const auto mat1_sizes = mat1.sizes();
  const auto mat2_sizes = mat2.sizes();
  auto is_transposed = [](const otter::IntArrayRef& strides, const otter::IntArrayRef& sizes) {
    return strides[1] == 1 && strides[2] >= sizes[1];
  };
  // gemm expects fortran order matrices, so we swap argument order to transpose everything
  const auto transpose_a = is_transposed(mat2_strides, mat2_sizes);
  const auto transpose_b = is_transposed(mat1_strides, mat1_sizes);
  const int64_t batch_size = mat1_sizes[0];
  const int64_t m = result_sizes[2];
  const int64_t n = result_sizes[1];
  const int64_t k = mat2_sizes[1];
  const int64_t lda = mat2_strides[transpose_a ? 2 : 1];
  const int64_t ldb = mat1_strides[transpose_b ? 2 : 1];
  const int64_t ldc = result_strides[1];
  OTTER_DISPATCH_FLOATING_TYPES(result.scalar_type(), "baddbmm_with_gemm", [&] {
    const auto alpha = alpha_.to<scalar_t>();
    const auto beta = beta_.to<scalar_t>();
    otter::gemm_batched_with_stride(
        transpose_a ? TransposeType::Transpose : TransposeType::NoTranspose,
        transpose_b ? TransposeType::Transpose : TransposeType::NoTranspose,
        batch_size, m, n, k, alpha,
        mat2.data_ptr<scalar_t>(), lda, mat2_strides[0],
        mat1.data_ptr<scalar_t>(), ldb, mat1_strides[0],
        beta,
        result.data_ptr<scalar_t>(), ldc, result_strides[0]);
  });
}
// This tries to apply some optimizations to bmm/baddbmm:
// - When the operand size is small, computation are parallelized over the batch
//   dimension using OMP and naive matrix multiplication is applied.
// - When the operand size is larger than the threshold, if compiled with MKL, MKL's batch gemm is used.
// - Otherwise, we use a series of matrix multiplications.
// The threshold of 400 for the first has not been thoroughly benchmarked yet and may have room for further
// optimization, it likely depends on the characteristics of the CPU, MKL will be different from non-MKL etc.,
// but this seems to be a first starting point.
static inline void bmm_out_or_baddbmm_(const Tensor& self_or_result_, const Tensor& batch1, const Tensor& batch2, const Scalar& beta, const Scalar& alpha, bool is_bmm_out) {
  // is_bmm_out: true for bmm_out, false for baddbmm_
  // self_or_result is "self" for baddbmm_ and "result" for bmm_out
  Tensor& self_or_result = const_cast<Tensor&>(self_or_result_);
  const auto batch1_sizes = batch1.sizes();
  const auto batch2_sizes = batch2.sizes();
  int64_t bs = batch1_sizes[0];
  int64_t contraction_size = batch1_sizes[2];
  int64_t res_rows = batch1_sizes[1];
  int64_t res_cols = batch2_sizes[2];
  // handle pathological cases that blas may not like
  if (self_or_result.numel() == 0) {
    return;
  } else if (contraction_size == 0) {
    if (is_bmm_out || (beta.to<double>() == 0.0)) {
      self_or_result.zero_();
      return;
    } else {
      self_or_result.mul_(beta);
      return;
    }
  }
//  auto batch_items_contiguous_or_transposed = [&](const Tensor& t) {
//    const auto sizes = t.sizes();
//    const auto strides = t.strides();
//    return (strides[2] == 1 && strides[1] >= sizes[2])
//            || (strides[1] == 1 && strides[2] >= sizes[1]);
//  };
  if (contraction_size * res_rows * res_cols < 400) {
    if (is_bmm_out) {
      OTTER_DISPATCH_ALL_TYPES(batch1.scalar_type(), "bmm", [&] {
          baddbmm_cpu_kernel<scalar_t, true>(self_or_result, batch1, batch2, beta, alpha);
        });
    } else {
      OTTER_DISPATCH_ALL_TYPES(batch1.scalar_type(), "baddbmm", [&] {
          baddbmm_cpu_kernel<scalar_t, false>(self_or_result, batch1, batch2, beta, alpha);
        });
    }
  } else { // split along batch dimension
#ifdef OTTER_MOBILE
    /*
     * We only do multithreading when Inference mode is enabled because various
     * thread local state is not appropriately propagated through
     * otter::parallel_for. e.g. RecordFunction related state, dispatchKeySet Big
     * concern with this is that if we use otter::parallel_for where state is not
     * propagated then dispatch machinery may work differently on main thread
     * vs. other threads, leading to undefined behavior.
     * Thus it is recommended to not use otter::parallel_for where lambdas do
     * ops that go through dispatcher.
     * For now we circument this by InferenceMode guard in order to unlock
     * performance.
     * Longer term we probably want a separate API that explicitly calls out
     * the TLS that it propagates.
     * Also note that this is enabled for mobile only because blas
     * implementation for non-mobile build is already multithreaded.
     */
    // Benchmarking was done as follows:
    // bmm_test: operator benchmark under
    // benchmarks/operator_benchmarks/pt/bmm_test.py Ran this benchmark for
    // various matrix sizes on Samsung S8U
    const bool enable_multithreaded_bmm = bs >= 4 && res_rows >= 4 && res_cols >= 16 && contraction_size >= 16;
#else
    const bool enable_multithreaded_bmm{false};
#endif
    if (is_bmm_out) {
      if (enable_multithreaded_bmm) {
        auto bmm_out_fn = [&](uint64_t start, uint64_t end) {
          for (const auto b : otter::irange(start, end)) {
            auto r = self_or_result.select(0, b);
            addmm_impl_cpu_(
                r, r, batch1.select(0, b), batch2.select(0, b), 0, 1);
          }
        };
        otter::parallel_for(0, bs, 1, bmm_out_fn);
      } else {
        for (const auto b : otter::irange(bs)) {
          auto r = self_or_result.select(0, b);
          addmm_impl_cpu_(r, r, batch1.select(0, b), batch2.select(0, b), 0, 1);
        }
      }
    } else {
      if (enable_multithreaded_bmm) {
        auto bmm_fn = [&](uint64_t start, uint64_t end) {
          for (const auto b : otter::irange(start, end)) {
            self_or_result.select(0, b).addmm_(
                batch1.select(0, b), batch2.select(0, b), beta, alpha);
          }
        };
        otter::parallel_for(0, bs, 1, bmm_fn);
      } else {
        for (const auto b : otter::irange(bs)) {
          self_or_result.select(0, b).addmm_(
              batch1.select(0, b), batch2.select(0, b), beta, alpha);
        }
      }
    }
  }
  return;
}

DEFINE_IMPL_FUNCTION(baddbmm_out_cpu)
(const Tensor & /*self*/, const Tensor & batch1, const Tensor & batch2, const Scalar& beta, const Scalar& alpha, const Tensor& result) {
    bmm_out_or_baddbmm_(result, batch1, batch2, beta, alpha, false);
  }
DEFINE_IMPL_FUNCTION(bmm_out_cpu)
(const Tensor & batch1, const Tensor & batch2, const Tensor & result) {
    {
    bmm_out_or_baddbmm_(result, batch1, batch2, Scalar(0.0), Scalar(1.0), true);
    }
}

#define det2(m)   ((double)m(0,0)*m(1,1) - (double)m(0,1)*m(1,0))
#define det3(m)   (m(0,0)*((double)m(1,1)*m(2,2) - (double)m(1,2)*m(2,1)) -  \
                   m(0,1)*((double)m(1,0)*m(2,2) - (double)m(1,2)*m(2,0)) +  \
                   m(0,2)*((double)m(1,0)*m(2,1) - (double)m(1,1)*m(2,0)))

#define Sf( y, x ) ((float*)(srcdata + y*srcstep))[x]
#define Sd( y, x ) ((double*)(srcdata + y*srcstep))[x]
#define Df( y, x ) ((float*)(dstdata + y*dststep))[x]
#define Dd( y, x ) ((double*)(dstdata + y*dststep))[x]

void mulTransposed(const Tensor& src, Tensor& dst, bool aTa, const Tensor& delta = otter::Tensor(), Scalar scale = 0, otter::ScalarType dtype = otter::ScalarType::Float) {
    if (delta.defined()) {
        otter::Tensor temp = src - delta;
        if (aTa)
            otter::native::mm_out(dst, temp, temp.transpose(0, 1));
        else
            otter::native::mm_out(dst, temp.transpose(0, 1), temp);
    } else {
        if (aTa)
            otter::native::mm_out(dst, src, src.transpose(0, 1));
        else
            otter::native::mm_out(dst, src.transpose(0, 1), src);
    }
    if (scale.toDouble())
        dst *= scale;
}

void transpose(const Tensor& src, Tensor& dst) {
    dst.copy_(src.transpose(0, 1));
}

void JacobiSVDImpl_m(const Tensor& _At, const Tensor& _W, const Tensor& _Vt) {
    OTTER_DISPATCH_FLOATING_TYPES(_At.scalar_type(), "cpu_svd", [&] {
        auto At = _At.accessor<scalar_t, 2>();
        auto Vt = _Vt.accessor<scalar_t, 2>();
        
        double minval = FLT_MIN;
        scalar_t eps = (scalar_t)(FLT_EPSILON * 2);
        const int m = _At.size(1);
        const int n = _W.size(0);
        const int n1 = m;
        AutoBuffer<double> W(n, 0);
        
        for (int i = 0; i < n; ++i) {
            double sd{0.};
            for (int k = 0; k < m; ++k) {
                scalar_t t = At[i][k];
                sd += (double)t * t;
            }
            W[i] = sd;
            
            for (int k = 0; k < n; ++k)
                Vt[i][k] = 0;
            Vt[i][i] = 1;
        }
        
        int max_iter = std::max(m, 30);
        for (int iter = 0; iter < max_iter; ++iter) {
            bool changed = false;
            scalar_t c, s;
            
            for (int i = 0; i < n - 1; ++i) {
                for (int j = i + 1; j < n; ++j) {
                    scalar_t* Ai = At[i].data(), *Aj = At[j].data();
                    double a = W[i], p = 0, b = W[j];
                    
                    for (int k = 0; k < m; ++k) {
                        p += (double)Ai[k] * Aj[k];
                    }
                    
                    if (std::abs(p) <= eps * std::sqrt((double)a * b))
                        continue;
                    
                    p *= 2;
                    double beta = a - b, gamma = hypot(p, beta);
                    if (beta < 0) {
                        double delta = (gamma - beta) * 0.5;
                        s = (scalar_t)std::sqrt(delta / gamma);
                        c = (scalar_t)(p / (gamma * s * 2));
                    } else {
                        c = (scalar_t)std::sqrt((gamma + beta) / (gamma * 2));
                        s = (scalar_t)(p / (gamma * c * 2));
                    }
                    
                    a = b = 0;
                    for (int k = 0; k < m; ++k) {
                        scalar_t t0 = c * Ai[k] + s * Aj[k];
                        scalar_t t1 = -s * Ai[k] + c * Aj[k];
                        Ai[k] = t0; Aj[k] = t1;
                        
                        a += (double)t0 * t0; b += (double)t1 * t1;
                    }
                    W[i] = a; W[j] = b;
                    
                    changed = true;
                    
                    scalar_t *Vi = Vt[i].data(), *Vj = Vt[j].data();
                    
                    for (int k = 0; k < n; ++k) {
                        scalar_t t0 = c * Vi[k] + s * Vj[k];
                        scalar_t t1 = -s * Vi[k] + c * Vj[k];
                        Vi[k] = t0; Vj[k] = t1;
                    }
                }
            }
            
            if (!changed)
                break;
        }
        
        for (int i = 0; i < n; ++i) {
            double sd{0.};
            
            for (int k = 0; k < m; ++k) {
                scalar_t t = At[i][k];
                sd += (double)t * t;
            }
            W[i] = std::sqrt(sd);
        }
        
        for (int i = 0; i < n - 1; ++i) {
            int j = i;
            for (int k = i + 1; k < n; ++k) {
                if (W[j] < W[k])
                    j = k;
            }
            if (i != j) {
                std::swap(W[i], W[j]);
                
                for (int k = 0; k < m; ++k)
                    std::swap(At[i][k], At[j][k]);
                
                for (int k = 0; k < n; ++k)
                    std::swap(Vt[i][k], Vt[j][k]);
            }
        }
        
        for (int i = 0; i < n; ++i)
            _W[i][0] = (scalar_t)W[i];
        
        srand(time(nullptr));
        
        for (int i = 0; i < n1; ++i) {
            double sd = i < n ? W[i] : 0;
            
            for (int ii = 0; ii < 100 & sd <= minval; ++ii) {
                const scalar_t val0 = (scalar_t)(1. / m);
                
                for (int k = 0; k < m; ++k) {
                    unsigned int rng = std::rand() % 4294967295; // 2 ^ 32 - 1
                    scalar_t val = (rng & 256) != 0 ? val0 : -val0;
                    At[i][k] = val;
                }
                
                for (int iter = 0; iter < 2; ++iter) {
                    for (int j = 0; j < i; ++j) {
                        sd = 0;
                        for (int k = 0; k < m; ++k)
                            sd += At[i][k] * At[j][k];
                        scalar_t asum = 0;
                        for (int k = 0; k < m; ++k) {
                            scalar_t t = (scalar_t)(At[i][k] - sd * At[j][k]);
                            At[i][k] = t;
                            asum += std::abs(t);
                        }
                        asum = asum > eps * 100 ? 1 / asum : 0;
                        for (int k = 0; k < n; ++k)
                            At[i][k] *= asum;
                    }
                }
                
                sd = 0;
                for (int k = 0; k < m; ++k) {
                    scalar_t t = At[i][k];
                    sd += (double)t * t;
                }
                sd = std::sqrt(sd);
            }
            
            scalar_t s = (scalar_t)(sd > minval ? 1 / sd : 0.);
            for (int k = 0; k < m; ++k)
                At[i][k] *= s;
        }
    });
}

int JacobiSVD_m(const Tensor& src, Tensor& matD, Tensor& matU, Tensor& matVt) {
    int m = src.size(0);
    int n = src.size(1);
    
    // check dimension
    
    bool at = false;
    if (m < n) {
        std::swap(m, n);
        at = true;
    }
    
    matD.resize_({n, 1});
    matD.zero_();
    matU.resize_({m, m});
    matU.zero_();
    matVt.resize_({n, n});
    matVt.zero_();
    
    auto tmp_u = matU, tmp_v = matVt;
    
    otter::Tensor tmp_a, tmp_a_;
    
    if (!at)
        tmp_a = src.transpose(0, 1);
    else
        tmp_a = src;
    
    if (m == n) {
        tmp_a_ = tmp_a;
    } else {
        tmp_a_ = otter::empty({m, m}, src.scalar_type());
        for (int i = 0; i < n; ++i) {
            tmp_a_[i].copy_(tmp_a[i]);
        }
    }
    JacobiSVDImpl_m(tmp_a_, matD, tmp_v);
    
    if (!at) {
        matVt = tmp_v;
        matU = tmp_a_;
    } else {
        matVt.copy_(tmp_v.transpose(0, 1));
        matVt = tmp_a_;
    }
    
    return 0;
}

template<typename _Tp> static inline _Tp hypot(_Tp a, _Tp b) {
    a = std::abs(a);
    b = std::abs(b);
    if (a > b) {
        b /= a;
        return a*std::sqrt(1 + b * b);
    }
    if (b > 0) {
        a /= b;
        return b*std::sqrt(1 + a * a);
    }
    return 0;
}

template<typename _Tp> bool
JacobiImpl_(_Tp* A, size_t astep, _Tp* W, _Tp* V, size_t vstep, int n, unsigned char* buf) {
    const _Tp eps = std::numeric_limits<_Tp>::epsilon();
    int i, j, k, m;

    astep /= sizeof(A[0]);
    if(V) {
        vstep /= sizeof(V[0]);
        for(i = 0; i < n; i++) {
            for(j = 0; j < n; j++)
                V[i*vstep + j] = (_Tp)0;
            V[i*vstep + i] = (_Tp)1;
        }
    }

    int iters, maxIters = n*n*30;

    int* indR = (int*)buf;
    int* indC = indR + n;
    _Tp mv = (_Tp)0;

    for(k = 0; k < n; k++) {
        W[k] = A[(astep + 1) * k];
        if(k < n - 1) {
            for( m = k + 1, mv = std::abs(A[astep * k + m]), i = k+2; i < n; i++) {
                _Tp val = std::abs(A[astep * k + i]);
                if(mv < val)
                    mv = val, m = i;
            }
            indR[k] = m;
        }
        if(k > 0)
        {
            for( m = 0, mv = std::abs(A[k]), i = 1; i < k; i++ )
            {
                _Tp val = std::abs(A[astep*i+k]);
                if( mv < val )
                    mv = val, m = i;
            }
            indC[k] = m;
        }
    }

    if( n > 1 ) for( iters = 0; iters < maxIters; iters++ )
    {
        // find index (k,l) of pivot p
        for( k = 0, mv = std::abs(A[indR[0]]), i = 1; i < n-1; i++ )
        {
            _Tp val = std::abs(A[astep*i + indR[i]]);
            if( mv < val )
                mv = val, k = i;
        }
        int l = indR[k];
        for( i = 1; i < n; i++ )
        {
            _Tp val = std::abs(A[astep*indC[i] + i]);
            if( mv < val )
                mv = val, k = indC[i], l = i;
        }

        _Tp p = A[astep*k + l];
        if( std::abs(p) <= eps )
            break;
        _Tp y = (_Tp)((W[l] - W[k])*0.5);
        _Tp t = std::abs(y) + hypot(p, y);
        _Tp s = hypot(p, t);
        _Tp c = t/s;
        s = p/s; t = (p/t)*p;
        if( y < 0 )
            s = -s, t = -t;
        A[astep*k + l] = 0;

        W[k] -= t;
        W[l] += t;

        _Tp a0, b0;

#undef rotate
#define rotate(v0, v1) a0 = v0, b0 = v1, v0 = a0 * c - b0 * s, v1 = a0 * s + b0 * c

        // rotate rows and columns k and l
        for (i = 0; i < k; i++)
            rotate(A[astep * i + k], A[astep * i + l]);
        for (i = k + 1; i < l; i++)
            rotate(A[astep * k + i], A[astep * i + l]);
        for (i = l + 1; i < n; i++)
            rotate(A[astep * k + i], A[astep * l + i]);

        // rotate eigenvectors
        if (V)
            for (i = 0; i < n; i++)
                rotate(V[vstep * k + i], V[vstep * l + i]);

#undef rotate

        for (j = 0; j < 2; j++) {
            int idx = j == 0 ? k : l;
            if (idx < n - 1) {
                for(m = idx + 1, mv = std::abs(A[astep * idx + m]), i = idx + 2; i < n; i++) {
                    _Tp val = std::abs(A[astep * idx + i]);
                    if (mv < val)
                        mv = val, m = i;
                }
                indR[idx] = m;
            }
            if (idx > 0) {
                for(m = 0, mv = std::abs(A[idx]), i = 1; i < idx; i++) {
                    _Tp val = std::abs(A[astep * i + idx]);
                    if (mv < val)
                        mv = val, m = i;
                }
                indC[idx] = m;
            }
        }
    }

    // sort eigenvalues & eigenvectors
    for (k = 0; k < n-1; k++) {
        m = k;
        for (i = k + 1; i < n; i++) {
            if (W[m] < W[i])
                m = i;
        }
        if (k != m) {
            std::swap(W[m], W[k]);
            if (V)
                for (i = 0; i < n; i++)
                    std::swap(V[vstep * m + i], V[vstep * k + i]);
        }
    }

    return true;
}

static bool Jacobi(float* S, size_t sstep, float* e, float* E, size_t estep, int n, unsigned char* buf) {
    return JacobiImpl_(S, sstep, e, E, estep, n, buf);
}

static bool Jacobi(double* S, size_t sstep, double* e, double* E, size_t estep, int n, unsigned char* buf) {
    return JacobiImpl_(S, sstep, e, E, estep, n, buf);
}


template<typename T> struct VBLAS {
    int dot(const T*, const T*, int, T*) const { return 0; }
    int givens(T*, T*, int, T, T) const { return 0; }
    int givensx(T*, T*, int, T, T, T*, T*) const { return 0; }
};

#if CV_SIMD
template<> inline int VBLAS<float>::dot(const float* a, const float* b, int n, float* result) const
{
    if( n < 2*v_float32::nlanes )
        return 0;
    int k = 0;
    v_float32 s0 = vx_setzero_f32();
    for( ; k <= n - v_float32::nlanes; k += v_float32::nlanes )
    {
        v_float32 a0 = vx_load(a + k);
        v_float32 b0 = vx_load(b + k);

        s0 += a0 * b0;
    }
    *result = v_reduce_sum(s0);
    vx_cleanup();
    return k;
}


template<> inline int VBLAS<float>::givens(float* a, float* b, int n, float c, float s) const
{
    if( n < v_float32::nlanes)
        return 0;
    int k = 0;
    v_float32 c4 = vx_setall_f32(c), s4 = vx_setall_f32(s);
    for( ; k <= n - v_float32::nlanes; k += v_float32::nlanes )
    {
        v_float32 a0 = vx_load(a + k);
        v_float32 b0 = vx_load(b + k);
        v_float32 t0 = (a0 * c4) + (b0 * s4);
        v_float32 t1 = (b0 * c4) - (a0 * s4);
        v_store(a + k, t0);
        v_store(b + k, t1);
    }
    vx_cleanup();
    return k;
}


template<> inline int VBLAS<float>::givensx(float* a, float* b, int n, float c, float s,
                                             float* anorm, float* bnorm) const
{
    if( n < v_float32::nlanes)
        return 0;
    int k = 0;
    v_float32 c4 = vx_setall_f32(c), s4 = vx_setall_f32(s);
    v_float32 sa = vx_setzero_f32(), sb = vx_setzero_f32();
    for( ; k <= n - v_float32::nlanes; k += v_float32::nlanes )
    {
        v_float32 a0 = vx_load(a + k);
        v_float32 b0 = vx_load(b + k);
        v_float32 t0 = (a0 * c4) + (b0 * s4);
        v_float32 t1 = (b0 * c4) - (a0 * s4);
        v_store(a + k, t0);
        v_store(b + k, t1);
        sa += t0 + t0;
        sb += t1 + t1;
    }
    *anorm = v_reduce_sum(sa);
    *bnorm = v_reduce_sum(sb);
    vx_cleanup();
    return k;
}

#if CV_SIMD_64F
template<> inline int VBLAS<double>::dot(const double* a, const double* b, int n, double* result) const
{
    if( n < 2*v_float64::nlanes )
        return 0;
    int k = 0;
    v_float64 s0 = vx_setzero_f64();
    for( ; k <= n - v_float64::nlanes; k += v_float64::nlanes )
    {
        v_float64 a0 = vx_load(a + k);
        v_float64 b0 = vx_load(b + k);

        s0 += a0 * b0;
    }
    double sbuf[2];
    v_store(sbuf, s0);
    *result = sbuf[0] + sbuf[1];
    vx_cleanup();
    return k;
}


template<> inline int VBLAS<double>::givens(double* a, double* b, int n, double c, double s) const
{
    int k = 0;
    v_float64 c2 = vx_setall_f64(c), s2 = vx_setall_f64(s);
    for( ; k <= n - v_float64::nlanes; k += v_float64::nlanes )
    {
        v_float64 a0 = vx_load(a + k);
        v_float64 b0 = vx_load(b + k);
        v_float64 t0 = (a0 * c2) + (b0 * s2);
        v_float64 t1 = (b0 * c2) - (a0 * s2);
        v_store(a + k, t0);
        v_store(b + k, t1);
    }
    vx_cleanup();
    return k;
}


template<> inline int VBLAS<double>::givensx(double* a, double* b, int n, double c, double s,
                                              double* anorm, double* bnorm) const
{
    int k = 0;
    v_float64 c2 = vx_setall_f64(c), s2 = vx_setall_f64(s);
    v_float64 sa = vx_setzero_f64(), sb = vx_setzero_f64();
    for( ; k <= n - v_float64::nlanes; k += v_float64::nlanes )
    {
        v_float64 a0 = vx_load(a + k);
        v_float64 b0 = vx_load(b + k);
        v_float64 t0 = (a0 * c2) + (b0 * s2);
        v_float64 t1 = (b0 * c2) - (a0 * s2);
        v_store(a + k, t0);
        v_store(b + k, t1);
        sa += t0 * t0;
        sb += t1 * t1;
    }
    double abuf[2], bbuf[2];
    v_store(abuf, sa);
    v_store(bbuf, sb);
    *anorm = abuf[0] + abuf[1];
    *bnorm = bbuf[0] + bbuf[1];
    return k;
}
#endif //CV_SIMD_64F
#endif //CV_SIMD

template<typename _Tp> void
JacobiSVDImpl_(_Tp* At, size_t astep, _Tp* _W, _Tp* Vt, size_t vstep,
               int m, int n, int n1, double minval, _Tp eps)
{
    VBLAS<_Tp> vblas;
    AutoBuffer<double> Wbuf(n);
    double* W = Wbuf.data();
    int i, j, k, iter, max_iter = std::max(m, 30);
    _Tp c, s;
    double sd;
    astep /= sizeof(At[0]);
    vstep /= sizeof(Vt[0]);

    for( i = 0; i < n; i++ )
    {
        for( k = 0, sd = 0; k < m; k++ )
        {
            _Tp t = At[i*astep + k];
            sd += (double)t*t;
        }
        W[i] = sd;

        if( Vt )
        {
            for( k = 0; k < n; k++ )
                Vt[i*vstep + k] = 0;
            Vt[i*vstep + i] = 1;
        }
    }

    for( iter = 0; iter < max_iter; iter++ )
    {
        bool changed = false;

        for( i = 0; i < n-1; i++ )
            for( j = i+1; j < n; j++ )
            {
                _Tp *Ai = At + i*astep, *Aj = At + j*astep;
                double a = W[i], p = 0, b = W[j];

                for( k = 0; k < m; k++ )
                    p += (double)Ai[k]*Aj[k];

                if( std::abs(p) <= eps*std::sqrt((double)a*b) )
                    continue;

                p *= 2;
                double beta = a - b, gamma = hypot((double)p, beta);
                if( beta < 0 )
                {
                    double delta = (gamma - beta)*0.5;
                    s = (_Tp)std::sqrt(delta/gamma);
                    c = (_Tp)(p/(gamma*s*2));
                }
                else
                {
                    c = (_Tp)std::sqrt((gamma + beta)/(gamma*2));
                    s = (_Tp)(p/(gamma*c*2));
                }

                a = b = 0;
                for( k = 0; k < m; k++ )
                {
                    _Tp t0 = c*Ai[k] + s*Aj[k];
                    _Tp t1 = -s*Ai[k] + c*Aj[k];
                    Ai[k] = t0; Aj[k] = t1;

                    a += (double)t0*t0; b += (double)t1*t1;
                }
                W[i] = a; W[j] = b;

                changed = true;

                if( Vt )
                {
                    _Tp *Vi = Vt + i*vstep, *Vj = Vt + j*vstep;
                    k = vblas.givens(Vi, Vj, n, c, s);

                    for( ; k < n; k++ )
                    {
                        _Tp t0 = c*Vi[k] + s*Vj[k];
                        _Tp t1 = -s*Vi[k] + c*Vj[k];
                        Vi[k] = t0; Vj[k] = t1;
                    }
                }
            }
        if( !changed )
            break;
    }

    for( i = 0; i < n; i++ )
    {
        for( k = 0, sd = 0; k < m; k++ )
        {
            _Tp t = At[i*astep + k];
            sd += (double)t*t;
        }
        W[i] = std::sqrt(sd);
    }

    for( i = 0; i < n-1; i++ )
    {
        j = i;
        for( k = i+1; k < n; k++ )
        {
            if( W[j] < W[k] )
                j = k;
        }
        if( i != j )
        {
            std::swap(W[i], W[j]);
            if( Vt )
            {
                for( k = 0; k < m; k++ )
                    std::swap(At[i*astep + k], At[j*astep + k]);

                for( k = 0; k < n; k++ )
                    std::swap(Vt[i*vstep + k], Vt[j*vstep + k]);
            }
        }
    }

    for( i = 0; i < n; i++ )
        _W[i] = (_Tp)W[i];

    if( !Vt )
        return;

    srand(0x12345678);
//    RNG rng(0x12345678);
    for( i = 0; i < n1; i++ )
    {
        sd = i < n ? W[i] : 0;

        for( int ii = 0; ii < 100 && sd <= minval; ii++ )
        {
            // if we got a zero singular value, then in order to get the corresponding left singular vector
            // we generate a random vector, project it to the previously computed left singular vectors,
            // subtract the projection and normalize the difference.
            const _Tp val0 = (_Tp)(1./m);
            for( k = 0; k < m; k++ )
            {
                _Tp val = (std::rand() & 256) != 0 ? val0 : -val0;
                At[i*astep + k] = val;
            }
            for( iter = 0; iter < 2; iter++ )
            {
                for( j = 0; j < i; j++ )
                {
                    sd = 0;
                    for( k = 0; k < m; k++ )
                        sd += At[i*astep + k]*At[j*astep + k];
                    _Tp asum = 0;
                    for( k = 0; k < m; k++ )
                    {
                        _Tp t = (_Tp)(At[i*astep + k] - sd*At[j*astep + k]);
                        At[i*astep + k] = t;
                        asum += std::abs(t);
                    }
                    asum = asum > eps*100 ? 1/asum : 0;
                    for( k = 0; k < m; k++ )
                        At[i*astep + k] *= asum;
                }
            }
            sd = 0;
            for( k = 0; k < m; k++ )
            {
                _Tp t = At[i*astep + k];
                sd += (double)t*t;
            }
            sd = std::sqrt(sd);
        }

        s = (_Tp)(sd > minval ? 1/sd : 0.);
        for( k = 0; k < m; k++ )
            At[i*astep + k] *= s;
    }
}

void SVD32f(float* At, size_t astep, float* W, float* /*U*/, size_t /*ustep*/, float* Vt, size_t vstep, int m, int n, int n1)
{
    JacobiSVDImpl_(At, astep, W, Vt, vstep, m, n, !Vt ? 0 : n1 < 0 ? n : n1, FLT_MIN, FLT_EPSILON*2);
}

void SVD64f(double* At, size_t astep, double* W, double* /*U*/, size_t /*ustep*/, double* Vt, size_t vstep, int m, int n, int n1)
{
    JacobiSVDImpl_(At, astep, W, Vt, vstep, m, n, !Vt ? 0 : n1 < 0 ? n : n1, DBL_MIN, DBL_EPSILON*10);
}

static void JacobiSVD(float* At, size_t astep, float* W, float* Vt, size_t vstep, int m, int n, int n1=-1)
{
    SVD32f(At, astep, W, NULL, astep, Vt, vstep, m, n, n1);
}

static void JacobiSVD(double* At, size_t astep, double* W, double* Vt, size_t vstep, int m, int n, int n1=-1)
{
    SVD64f(At, astep, W, NULL, astep, Vt, vstep, m, n, n1);
}

template<typename T1, typename T2, typename T3> static void
MatrAXPY( int m, int n, const T1* x, int dx,
         const T2* a, int inca, T3* y, int dy )
{
    int i;
    for( i = 0; i < m; i++, x += dx, y += dy )
    {
        T2 s = a[i*inca];
        int j = 0;
         #if CV_ENABLE_UNROLLED
        for(; j <= n - 4; j += 4 )
        {
            T3 t0 = (T3)(y[j]   + s*x[j]);
            T3 t1 = (T3)(y[j+1] + s*x[j+1]);
            y[j]   = t0;
            y[j+1] = t1;
            t0 = (T3)(y[j+2] + s*x[j+2]);
            t1 = (T3)(y[j+3] + s*x[j+3]);
            y[j+2] = t0;
            y[j+3] = t1;
        }
        #endif
        for( ; j < n; j++ )
            y[j] = (T3)(y[j] + s*x[j]);
    }
}

template<typename T>
static void SVBkSbImpl_(int m, int n, const T* w, int incw,
       const T* u, int ldu, bool uT,
       const T* v, int ldv, bool vT,
       const T* b, int ldb, int nb,
       T* x, int ldx, double* buffer, T eps) {
    double threshold = 0;
    int udelta0 = uT ? ldu : 1, udelta1 = uT ? 1 : ldu;
    int vdelta0 = vT ? ldv : 1, vdelta1 = vT ? 1 : ldv;
    int i, j, nm = std::min(m, n);

    if (!b)
        nb = m;

    for (i = 0; i < n; i++)
        for (j = 0; j < nb; j++)
            x[i * ldx + j] = 0;

    for (i = 0; i < nm; i++)
        threshold += w[i * incw];
    threshold *= eps;

    // v * inv(w) * uT * b
    for (i = 0; i < nm; i++, u += udelta0, v += vdelta0) {
        double wi = w[i * incw];
        if ((double)std::abs(wi) <= threshold)
            continue;
        wi = 1 / wi;

        if (nb == 1) {
            double s = 0;
            if (b)
                for (j = 0; j < m; j++)
                    s += u[j * udelta1] * b[j * ldb];
            else
                s = u[0];
            s *= wi;

            for (j = 0; j < n; j++)
                x[j * ldx] = (T)(x[j * ldx] + s * v[j * vdelta1]);
        } else {
            if (b) {
                for (j = 0; j < nb; j++)
                    buffer[j] = 0;
                MatrAXPY(m, nb, b, ldb, u, udelta1, buffer, 0);
                for (j = 0; j < nb; j++)
                    buffer[j] *= wi;
            } else {
                for (j = 0; j < nb; j++)
                    buffer[j] = u[j * udelta1] * wi;
            }
            MatrAXPY(n, nb, buffer, 0, v, vdelta1, x, ldx);
        }
    }
}

static void SVBkSb (int m, int n, const float* w, size_t wstep,
        const float* u, size_t ustep, bool uT,
        const float* v, size_t vstep, bool vT,
        const float* b, size_t bstep, int nb,
        float* x, size_t xstep, unsigned char* buffer) {
    SVBkSbImpl_(m, n, w, wstep ? (int)(wstep/sizeof(w[0])) : 1,
                u, (int)(ustep/sizeof(u[0])), uT,
                v, (int)(vstep/sizeof(v[0])), vT,
                b, (int)(bstep/sizeof(b[0])), nb,
                x, (int)(xstep/sizeof(x[0])),
                (double*)buffer, (float)(DBL_EPSILON*2) );
}

static void SVBkSb(int m, int n, const double* w, size_t wstep,
       const double* u, size_t ustep, bool uT,
       const double* v, size_t vstep, bool vT,
       const double* b, size_t bstep, int nb,
       double* x, size_t xstep, unsigned char* buffer) {
    SVBkSbImpl_(m, n, w, wstep ? (int)(wstep/sizeof(w[0])) : 1,
                u, (int)(ustep/sizeof(u[0])), uT,
                v, (int)(vstep/sizeof(v[0])), vT,
                b, (int)(bstep/sizeof(b[0])), nb,
                x, (int)(xstep/sizeof(x[0])),
                (double*)buffer, DBL_EPSILON*2 );
}

bool solve(const Tensor& src, const Tensor& _src2, Tensor& _dst, int method) {
    bool result = true;
    ScalarType type = src.scalar_type();
    bool is_normal = (method & DECOMP_NORMAL) != 0;
    
    OTTER_CHECK(type == _src2.scalar_type() && (type == ScalarType::Float || type == ScalarType::Double), "Require two type is same and type is float or double");
    
    method &= ~DECOMP_NORMAL;
    
    OTTER_CHECK(method == DECOMP_LU || method == DECOMP_SVD || method == DECOMP_EIG || method == DECOMP_CHOLESKY || method == DECOMP_QR, "Unsupported method, see #DecompTypes");
    OTTER_CHECK((method != DECOMP_LU && method != DECOMP_CHOLESKY) || is_normal || src.size(0) == src.size(1), "Require Square matrix");
    
    // check case of a single equation and small matrix
    if( (method == DECOMP_LU || method == DECOMP_CHOLESKY) && !is_normal &&
        src.size(0) <= 3 && src.size(0) == src.size(1) && _src2.size(1) == 1 ) {
        _dst = otter::empty({src.size(1), _src2.size(1)}, src.scalar_type());
        Tensor dst = _dst;

        #define bf(y) ((float*)(bdata + y*src2step))[0]
        #define bd(y) ((double*)(bdata + y*src2step))[0]

        const unsigned char* srcdata = src.data_ptr<unsigned char>();
        const unsigned char* bdata = _src2.data_ptr<unsigned char>();
        unsigned char* dstdata = dst.data_ptr<unsigned char>();
        size_t srcstep = src.itemsize() * src.size(1);
        size_t src2step = _src2.itemsize() * _src2.size(1);
        size_t dststep = dst.itemsize() * dst.size(1);

        if (src.size(0) == 2 ) {
            if (type == otter::ScalarType::Float) {
                double d = det2(Sf);
                if (d != 0.) {
                    double t;
                    d = 1./d;
                    t = (float)(((double)bf(0)*Sf(1,1) - (double)bf(1)*Sf(0,1))*d);
                    Df(1,0) = (float)(((double)bf(1)*Sf(0,0) - (double)bf(0)*Sf(1,0))*d);
                    Df(0,0) = (float)t;
                } else
                    result = false;
            }
            else {
                double d = det2(Sd);
                if( d != 0. ) {
                    double t;
                    d = 1./d;
                    t = (bd(0)*Sd(1,1) - bd(1)*Sd(0,1))*d;
                    Dd(1,0) = (bd(1)*Sd(0,0) - bd(0)*Sd(1,0))*d;
                    Dd(0,0) = t;
                } else
                    result = false;
            }
        } else if (src.size(0) == 3) {
            if(type == ScalarType::Float) {
                double d = det3(Sf);
                if (d != 0.)  {
                    float t[3];
                    d = 1./d;

                    t[0] = (float)(d*
                           (bf(0)*((double)Sf(1,1)*Sf(2,2) - (double)Sf(1,2)*Sf(2,1)) -
                            Sf(0,1)*((double)bf(1)*Sf(2,2) - (double)Sf(1,2)*bf(2)) +
                            Sf(0,2)*((double)bf(1)*Sf(2,1) - (double)Sf(1,1)*bf(2))));

                    t[1] = (float)(d*
                           (Sf(0,0)*(double)(bf(1)*Sf(2,2) - (double)Sf(1,2)*bf(2)) -
                            bf(0)*((double)Sf(1,0)*Sf(2,2) - (double)Sf(1,2)*Sf(2,0)) +
                            Sf(0,2)*((double)Sf(1,0)*bf(2) - (double)bf(1)*Sf(2,0))));

                    t[2] = (float)(d*
                           (Sf(0,0)*((double)Sf(1,1)*bf(2) - (double)bf(1)*Sf(2,1)) -
                            Sf(0,1)*((double)Sf(1,0)*bf(2) - (double)bf(1)*Sf(2,0)) +
                            bf(0)*((double)Sf(1,0)*Sf(2,1) - (double)Sf(1,1)*Sf(2,0))));

                    Df(0,0) = t[0];
                    Df(1,0) = t[1];
                    Df(2,0) = t[2];
                } else
                    result = false;
            } else {
                double d = det3(Sd);
                if (d != 0.) {
                    double t[9];

                    d = 1./d;

                    t[0] = ((Sd(1,1) * Sd(2,2) - Sd(1,2) * Sd(2,1))*bd(0) +
                            (Sd(0,2) * Sd(2,1) - Sd(0,1) * Sd(2,2))*bd(1) +
                            (Sd(0,1) * Sd(1,2) - Sd(0,2) * Sd(1,1))*bd(2))*d;

                    t[1] = ((Sd(1,2) * Sd(2,0) - Sd(1,0) * Sd(2,2))*bd(0) +
                            (Sd(0,0) * Sd(2,2) - Sd(0,2) * Sd(2,0))*bd(1) +
                            (Sd(0,2) * Sd(1,0) - Sd(0,0) * Sd(1,2))*bd(2))*d;

                    t[2] = ((Sd(1,0) * Sd(2,1) - Sd(1,1) * Sd(2,0))*bd(0) +
                            (Sd(0,1) * Sd(2,0) - Sd(0,0) * Sd(2,1))*bd(1) +
                            (Sd(0,0) * Sd(1,1) - Sd(0,1) * Sd(1,0))*bd(2))*d;

                    Dd(0,0) = t[0];
                    Dd(1,0) = t[1];
                    Dd(2,0) = t[2];
                } else
                    result = false;
            }
        } else {
            OTTER_INTERNAL_ASSERT(src.size(0) == 1);

            if (type == otter::ScalarType::Float) {
                double d = Sf(0,0);
                if( d != 0. )
                    Df(0,0) = (float)(bf(0)/d);
                else
                    result = false;
            } else {
                double d = Sd(0,0);
                if( d != 0. )
                    Dd(0,0) = (bd(0)/d);
                else
                    result = false;
            }
        }
        return result;
    }

    int m = src.size(0), m_ = m, n = src.size(1), nb = _src2.size(1);
    size_t esz = src.itemsize(), bufsize = 0;
    size_t vstep = n * esz;
    size_t astep = (method == DECOMP_SVD && !is_normal) ? m * esz : vstep;
    otter::AutoBuffer<unsigned char> buffer;

    otter::Tensor src2 = _src2;
    _dst = otter::empty({src.size(1), src2.size(1)}, src.scalar_type());
    Tensor dst = _dst;

    if (m < n)
        OTTER_CHECK(false, "The function can not solve under-determined linear systems" );

    if (m == n)
        is_normal = false;
    else if (is_normal) {
        m_ = n;
        if(method == DECOMP_SVD)
            method = DECOMP_EIG;
    }

    size_t asize = astep * (method == DECOMP_SVD || is_normal ? n : m);
    bufsize += asize + 32;

    if (is_normal)
        bufsize += n * nb * esz;
    if (method == DECOMP_SVD || method == DECOMP_EIG)
        bufsize += n * 5 * esz + n * vstep + nb * sizeof(double) + 32;

    buffer.allocate(bufsize);
    unsigned char* ptr = buffer.data();

    otter::Tensor a = otter::from_blob(ptr, {m_, n}, src.scalar_type());

    if (is_normal)
        mulTransposed(src, a, true);
    else if (method != DECOMP_SVD)
        a.copy_(src);
    else {
        a = otter::from_blob(ptr, {n, m_}, type);
        transpose(src, a);
    }
    ptr += asize;

    if (!is_normal) {
        if (method == DECOMP_LU || method == DECOMP_CHOLESKY)
            dst.copy_(src2);
    } else {
        // a'*b
        if (method == DECOMP_LU || method == DECOMP_CHOLESKY)
            otter::native::addmm_out(dst, otter::Tensor(), src, src2, 0, 1);
        else {
            otter::Tensor tmp = otter::from_blob(ptr, {n, nb}, type);
            ptr += n * nb * esz;
            otter::native::addmm_out(tmp, otter::Tensor(), src, src2, 0, 1);
            src2 = tmp;
        }
    }

    if (method == DECOMP_LU) {
//        if( type == CV_32F )
//            result = hal::LU32f(a.ptr<float>(), a.step, n, dst.ptr<float>(), dst.step, nb) != 0;
//        else
//            result = hal::LU64f(a.ptr<double>(), a.step, n, dst.ptr<double>(), dst.step, nb) != 0;
    } else if(method == DECOMP_CHOLESKY) {
//        if( type == CV_32F )
//            result = hal::Cholesky32f(a.ptr<float>(), a.step, n, dst.ptr<float>(), dst.step, nb);
//        else
//            result = hal::Cholesky64f(a.ptr<double>(), a.step, n, dst.ptr<double>(), dst.step, nb);
    } else if(method == DECOMP_QR) {
//        Mat rhsMat;
//        if( is_normal || m == n )
//        {
//            src2.copyTo(dst);
//            rhsMat = dst;
//        }
//        else
//        {
//            rhsMat = Mat(m, nb, type);
//            src2.copyTo(rhsMat);
//        }
//
//        if( type == CV_32F )
//            result = hal::QR32f(a.ptr<float>(), a.step, a.size(0), a.size(1), rhsMat.size(1), rhsMat.ptr<float>(), rhsMat.step, NULL) != 0;
//        else
//            result = hal::QR64f(a.ptr<double>(), a.step, a.size(0), a.size(1), rhsMat.size(1), rhsMat.ptr<double>(), rhsMat.step, NULL) != 0;
//
//        if (rhsMat.size(0) != dst.size(0))
//            rhsMat.rowRange(0, dst.size(0)).copyTo(dst);
    } else {
        otter::Tensor v = otter::from_blob(ptr, {n, n}, type);
        otter::Tensor w = otter::from_blob(ptr + vstep * n, {n, 1}, type);
        otter::Tensor u;
        ptr += n * (vstep + esz);

        if (method == DECOMP_EIG) {
            if (type == otter::ScalarType::Float)
                Jacobi(a.data_ptr<float>(), (a.itemsize() * a.size(1)), w.data_ptr<float>(), v.data_ptr<float>(), (v.itemsize() * v.size(1)), n, ptr);
            else
                Jacobi(a.data_ptr<double>(), (a.itemsize() * a.size(1)), w.data_ptr<double>(), v.data_ptr<double>(), (v.itemsize() * v.size(1)), n, ptr);
            u = v;
        } else {
            if (type == otter::ScalarType::Float)
                JacobiSVD(a.data_ptr<float>(), (a.itemsize() * a.size(1)), w.data_ptr<float>(), v.data_ptr<float>(), (v.itemsize() * v.size(1)), m_, n);
            else
                JacobiSVD(a.data_ptr<double>(), (a.itemsize() * a.size(1)), w.data_ptr<double>(), v.data_ptr<double>(), (v.itemsize() * v.size(1)), m_, n);
            u = a;
        }

        if (type == otter::ScalarType::Float) {
            SVBkSb(m_, n, w.data_ptr<float>(), 0, u.data_ptr<float>(), (u.itemsize() * u.size(1)), true, v.data_ptr<float>(), (v.itemsize() * v.size(1)), true, src2.data_ptr<float>(), (src2.itemsize() * src2.size(1)), nb, dst.data_ptr<float>(), (dst.itemsize() * dst.size(1)), ptr);
        } else {
            SVBkSb(m_, n, w.data_ptr<double>(), 0, u.data_ptr<double>(), (u.itemsize() * u.size(1)), true, v.data_ptr<double>(), (v.itemsize() * v.size(1)), true, src2.data_ptr<double>(), (src2.itemsize() * src2.size(1)), nb, dst.data_ptr<double>(), (dst.itemsize() * dst.size(1)), ptr);
        }
        result = true;
    }

    if (!result)
        dst = Scalar(0);

    return result;
}

}   // end namespace otter
