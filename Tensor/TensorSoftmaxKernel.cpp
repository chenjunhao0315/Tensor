//
//  TensorSoftmaxKernel.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/5/26.
//

#include "Tensor.hpp"
#include "TensorSoftmax.hpp"
#include "VecFunctional.hpp"
#include "Parallel.hpp"
#include "Dispatch.hpp"

namespace otter {

constexpr int64_t GRAIN_SIZE = 32768;

template <typename scalar_t>
inline void _vec_softmax_lastdim(
    scalar_t* input_data_base,
    scalar_t* output_data_base,
    int64_t outer_size,
    int64_t dim_size) {
  using Vec = vec::Vectorized<scalar_t>;
  int64_t grain_size = std::max(GRAIN_SIZE / (16 * dim_size), (int64_t)1);
  parallel_for(0, outer_size, grain_size, [&](int64_t begin, int64_t end) {
    for (const auto i : otter::irange(begin, end)) {
      scalar_t* input_data = input_data_base + i * dim_size;
      scalar_t* output_data = output_data_base + i * dim_size;
      scalar_t max_input = vec::reduce_all<scalar_t>(
          [](Vec& x, Vec& y) { return vec::maximum(x, y); },
          input_data,
          dim_size);
      vec::map(
          [max_input](Vec x) { return (x - Vec(max_input)).exp(); },
          output_data,
          input_data,
          dim_size);
      scalar_t tmp_sum = vec::reduce_all<scalar_t>(
          [](Vec x, Vec y) { return x + y; }, output_data, dim_size);
      tmp_sum = 1 / tmp_sum;
      vec::map(
          [tmp_sum](Vec x) { return x * Vec(tmp_sum); },
          output_data,
          output_data,
          dim_size);
    }
  });
}

template <typename scalar_t>
struct vec_host_softmax_lastdim {
  static void apply(const Tensor& output, const Tensor& input) {
    int64_t outer_size = 1;
    int64_t dim_size = input.size(input.dim() - 1);
    for (int64_t i = 0; i < input.dim() - 1; ++i)
      outer_size *= input.size(i);
    scalar_t* input_data_base = input.data_ptr<scalar_t>();
    scalar_t* output_data_base = output.data_ptr<scalar_t>();
      
    _vec_softmax_lastdim(
          input_data_base, output_data_base, outer_size, dim_size);
  }
};

template <typename scalar_t>
inline void _vec_softmax(
    scalar_t* input_data_base,
    scalar_t* output_data_base,
    int64_t outer_size,
    int64_t inner_size,
    int64_t dim_size) {
  using Vec = vec::Vectorized<scalar_t>;
  int64_t dim_stride = inner_size;
  int64_t outer_stride = dim_size * dim_stride;
  int64_t grain_size = std::min(GRAIN_SIZE / dim_size, (int64_t)1);
  int vectorized_step = Vec().size();
  parallel_for(
      0, outer_size * inner_size, grain_size, [&](int64_t begin, int64_t end) {
        int64_t idx = begin;
        while (idx < end) {
          int64_t outer_idx = idx / inner_size;
          int64_t inner_idx = idx % inner_size;
          if (((inner_idx + vectorized_step) <= inner_size) && ((idx + vectorized_step) <= end)) {
            // Vectorization
            scalar_t* input_data =
                input_data_base + outer_idx * outer_stride + inner_idx;
            scalar_t* output_data =
                output_data_base + outer_idx * outer_stride + inner_idx;
            // Step 1: Get max Score
            Vec max_vec = Vec::loadu(input_data);
            for (const auto d : otter::irange(1, dim_size)) {
              Vec input_vec = Vec::loadu(input_data + d * dim_stride);
              max_vec = vec::maximum(max_vec, input_vec);
            }
            // Step2: Calculate sum
            Vec sum_vec = Vec(0.0);
            for (const auto d : otter::irange(dim_size)) {
              Vec output_vec =
                  (Vec::loadu(input_data + d * dim_stride) - max_vec).exp();
              output_vec.store(output_data + d * dim_stride);
              sum_vec = sum_vec + output_vec;
            }
            // Step3: Unify
            for (const auto d : otter::irange(dim_size)) {
              Vec output_vec =
                  Vec::loadu(output_data + d * dim_stride) / sum_vec;
              output_vec.store(output_data + d * dim_stride);
            }
            idx += vectorized_step;
          } else {
            // Tail case(Scalar): it is exactly same logic as host_softmax
            // inside aten/src/ATen/native/SoftMax.cpp. There are 2 kind of
            // cases which will fall through this part:
            // Case 1: For the idx at the end of total chunk for each thread, there are not enough numbers for parallization.
            // Case 2: For the idx at the end of each inner_size inside thread, there are not enough numbers for parallization.
            int64_t tail_number = ((idx+vectorized_step) > end) ? /*Case1*/ (end - idx) : /*Case2*/ (inner_size - inner_idx);
            for (const auto i : otter::irange(tail_number)) {
              outer_idx = (idx + i) / inner_size;
              inner_idx = (idx + i) % inner_size;
              scalar_t* input_data =
                  input_data_base + outer_idx * outer_stride + inner_idx;
              scalar_t* output_data =
                  output_data_base + outer_idx * outer_stride + inner_idx;
              // Step1: Get max score
              scalar_t max_input = input_data[0];
              for (const auto d : otter::irange(1, dim_size)) {
                max_input = std::max(max_input, input_data[d * dim_stride]);
              }
              // Step2: Calculate the Sum
              scalar_t sum_data = 0;
              for (const auto d : otter::irange(dim_size)) {
                output_data[d * dim_stride] =
                    std::exp(input_data[d * dim_stride] - max_input);
                sum_data += output_data[d * dim_stride];
              }
              // Step3: Unify
              for (const auto d : otter::irange(dim_size)) {
                output_data[d * dim_stride] =
                    output_data[d * dim_stride]/sum_data;
              }
            }
            idx += tail_number;
          }
        }
      });
}

template <typename scalar_t>
struct vec_softmax {
  static void apply(const Tensor& output, const Tensor& input, int64_t dim) {
    int64_t outer_size = 1;
    int64_t dim_size = input.size(dim);
    int64_t inner_size = 1;
    for (const auto i : otter::irange(dim))outer_size *= input.size(i);
    for (int64_t i = dim + 1; i < input.dim(); ++i)
      inner_size *= input.size(i);
    scalar_t* input_data_base = input.data_ptr<scalar_t>();
    scalar_t* output_data_base = output.data_ptr<scalar_t>();

    _vec_softmax(
        input_data_base, output_data_base, outer_size, inner_size, dim_size);

  }
};

static void softmax_lastdim_kernel_impl(
    const Tensor& result,
    const Tensor& self) {
  OTTER_DISPATCH_FLOATING_TYPES(self.scalar_type(),
      "softmax_lastdim_kernel_impl",
      [&] { vec_host_softmax_lastdim<scalar_t>::apply(result, self); });
}

static void softmax_kernel_impl(const Tensor& result, const Tensor& self, int64_t dim) {
  OTTER_DISPATCH_FLOATING_TYPES(self.scalar_type(),
    "softmax_kernel_impl",
    [&] { vec_softmax<scalar_t>::apply(result, self, dim); });
}

REGISTER_DISPATCH(softmax_lastdim_kernel, &softmax_lastdim_kernel_impl);
REGISTER_DISPATCH(softmax_kernel, &softmax_kernel_impl);


}   // end namespace otter
