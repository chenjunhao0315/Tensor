//
//  SortingKernel.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/7/11.
//

#include "Sorting.hpp"
#include "SortingKernel.hpp"
#include "Parallel.hpp"
#include "TensorIterator.hpp"
#include "Dispatch.hpp"
#include "Utils.hpp"
#include "Math.hpp"

namespace otter {

// Core topk loop, shared between CPU and QuantizedCPU
template <typename scalar_t, typename accscalar_t>
void topk_impl_loop(
                    const int64_t mode_values_stride,
                    const int64_t mode_indices_stride,
                    const int64_t tmp_values_stride,
                    const int64_t k,
                    const int64_t dim_size,
                    const bool largest,
                    const bool sorted,
                    char** data, const int64_t* strides, const int64_t n) {
    
    using elem_t = std::pair<accscalar_t, int64_t>;
    std::vector<elem_t> queue(dim_size);
    for (const auto i : otter::irange(n)) {
        TensorAccessor<scalar_t, 1> mode_values(
                                                reinterpret_cast<scalar_t*>(data[0] + i * strides[0]),
                                                &k, &mode_values_stride);
        TensorAccessor<int64_t, 1> mode_indices(
                                                reinterpret_cast<int64_t*>(data[1] + i * strides[1]),
                                                &k, &mode_indices_stride);
        TensorAccessor<scalar_t, 1> tmp_values(
                                               reinterpret_cast<scalar_t*>(data[2] + i * strides[2]),
                                               &dim_size, &tmp_values_stride);
        
        auto n = dim_size;
        auto use_partial_sort = k * 64 <= n;
        
        for (const auto j : otter::irange(n)) {
            queue[j].first = tmp_values[j];
            queue[j].second = j;
        }
        
        // we want nan to be sorted as top for numpy compatibility
        if (use_partial_sort) {
            if (largest) {
                std::partial_sort(queue.begin(), queue.begin() + k, queue.end(),
                                  [](const elem_t& x, const elem_t& y) -> bool {
                    return ((otter::_isnan<accscalar_t>(x.first) && !otter::_isnan<accscalar_t>(y.first)) || (x.first > y.first));
                });
            } else {
                std::partial_sort(queue.begin(), queue.begin() + k, queue.end(),
                                  [](const elem_t& x, const elem_t& y) -> bool {
                    return ((!otter::_isnan<accscalar_t>(x.first) && otter::_isnan<accscalar_t>(y.first)) || (x.first < y.first));
                });
            }
        } else {
            if (largest) {
                std::nth_element(queue.begin(), queue.begin() + k - 1, queue.end(),
                                 [](const elem_t& x, const elem_t& y) -> bool {
                    return ((otter::_isnan<accscalar_t>(x.first) && !otter::_isnan<accscalar_t>(y.first)) || (x.first > y.first));
                });
                if (sorted) {
                    std::sort(queue.begin(), queue.begin() + k - 1,
                              [](const elem_t& x, const elem_t& y) -> bool {
                        return ((otter::_isnan<accscalar_t>(x.first) && !otter::_isnan<accscalar_t>(y.first)) || (x.first > y.first));
                    });
                }
            } else {
                std::nth_element(queue.begin(), queue.begin() + k -1, queue.end(),
                                 [](const elem_t& x, const elem_t& y) -> bool {
                    return ((!otter::_isnan<accscalar_t>(x.first) && otter::_isnan<accscalar_t>(y.first)) || (x.first < y.first));
                });
                if (sorted) {
                    std::sort(queue.begin(), queue.begin() + k -1,
                              [](const elem_t& x, const elem_t& y) -> bool {
                        return ((!otter::_isnan<accscalar_t>(x.first) && otter::_isnan<accscalar_t>(y.first)) || (x.first < y.first));
                    });
                }
            }
        }
        
        for (const auto j : otter::irange(k)) {
            mode_values[j] = queue[j].first;
            mode_indices[j] = queue[j].second;
        }
    }
}


template <typename func_t>
void _dim_apply(
                const TensorBase &values,
                const TensorBase &indices,
                int64_t dim,
                const std::string& /*method_name*/,
                const func_t& f) {
    auto iter = TensorIteratorConfig()
        .check_all_same_dtype(false)
        .resize_outputs(false)
        .declare_static_shape(values.sizes(), /*squash_dims=*/dim)
        .add_output(values)
        .add_output(indices)
        .build();
    
    auto values_dim_stride = values.stride(dim);
    auto indices_dim_stride = indices.stride(dim);
    auto dim_size = values.size(dim);
    
    OTTER_DISPATCH_ALL_TYPES_AND(
                                 ScalarType::Bool, iter.dtype(),
                                 "sorting_kernel_method_name", [&] {
                                     auto loop = [&](char** data, const int64_t* strides, int64_t n) {
                                         auto* values_data_bytes = data[0];
                                         auto* indices_data_bytes = data[1];
                                         
                                         if(values_data_bytes==nullptr || indices_data_bytes==nullptr){
                                             return;
                                         }
                                         
                                         for (const auto i : otter::irange(n)) {
                                             (void)i; //Suppress unused variable warning
                                             f(
                                               reinterpret_cast<scalar_t*>(values_data_bytes),
                                               values_dim_stride,
                                               reinterpret_cast<int64_t*>(indices_data_bytes),
                                               indices_dim_stride,
                                               dim_size
                                               );
                                             
                                             values_data_bytes += strides[0];
                                             indices_data_bytes += strides[1];
                                         }
                                     };
                                     
                                     int64_t grain_size = otter::GRAIN_SIZE / std::max(int64_t{1}, dim_size);
                                     iter.for_each(loop, /*grain_size=*/grain_size);
                                 }
                                 );
}

template <typename scalar_t>
struct KeyValueCompAsc {
    template <typename LHS, typename RHS>
    constexpr bool operator()(LHS lhs, RHS rhs) const {
        return (!otter::_isnan<scalar_t>(get<0>(lhs)) && otter::_isnan<scalar_t>(get<0>(rhs)))
        || (get<0>(lhs) < get<0>(rhs));
    }
};

template <typename scalar_t>
struct KeyValueCompDesc {
    template <typename LHS, typename RHS>
    constexpr bool operator()(LHS lhs, RHS rhs) const {
        return (otter::_isnan<scalar_t>(get<0>(lhs)) && !otter::_isnan<scalar_t>(get<0>(rhs)))
        || (get<0>(lhs) > get<0>(rhs));
    }
};

static void sort_kernel(
                        const TensorBase& /*self*/,
                        const TensorBase& values,
                        const TensorBase& indices,
                        int64_t dim,
                        bool descending,
                        bool stable) {
    dim = maybe_wrap_dim(dim, values.dim());
    _fill_indices(indices, dim);
    _dim_apply(
               values, indices, dim,
               "sort_cpu", [&](
                               auto* values, int64_t values_dim_stride,
                               auto* indices, int64_t indices_dim_stride,
                               int64_t dim_size
                               ) {
                                   using scalar_t = typename std::remove_pointer<decltype(values)>::type;
                                   auto values_accessor = StridedRandomAccessor<scalar_t>(
                                                                                          values, values_dim_stride);
                                   auto indices_accessor = StridedRandomAccessor<int64_t>(
                                                                                          indices, indices_dim_stride);
                                   auto composite_accessor = CompositeRandomAccessorCPU<
                                   decltype(values_accessor), decltype(indices_accessor)
                                   >(values_accessor, indices_accessor);
                                   
                                   if (descending) {
                                       if (stable) {
                                           std::stable_sort(composite_accessor, composite_accessor + dim_size,
                                                            KeyValueCompDesc<scalar_t>());
                                       }
                                       else {
                                           std::sort(composite_accessor, composite_accessor + dim_size,
                                                     KeyValueCompDesc<scalar_t>());
                                       }
                                   }
                                   else {
                                       if (stable) {
                                           std::stable_sort(composite_accessor, composite_accessor + dim_size,
                                                            KeyValueCompAsc<scalar_t>());
                                       }
                                       else {
                                           std::sort(composite_accessor, composite_accessor + dim_size,
                                                     KeyValueCompAsc<scalar_t>());
                                       }
                                   }
                               }
               );
}

static void topk_kernel(
                        const TensorBase &values,
                        const TensorBase &indices,
                        const TensorBase &self,
                        int64_t k,
                        int64_t dim,
                        bool largest,
                        bool sorted) {
    auto sizes = self.sizes();
    auto iter = TensorIteratorConfig()
        .check_all_same_dtype(false)
        .resize_outputs(false)
        .declare_static_shape(sizes, /*squash_dims=*/dim)
        .add_output(values)
        .add_output(indices)
        .add_input(self)
        .build();
    
    auto mode_values_stride = values.strides()[dim];
    auto mode_indices_stride = indices.strides()[dim];
    auto tmp_values_stride = self.strides()[dim];
    
    OTTER_DISPATCH_ALL_TYPES(self.scalar_type(), "topk_cpu", [&] {
        auto loop = [&](char** data, const int64_t* strides, int64_t n) {
            return topk_impl_loop<scalar_t, scalar_t>(
                                                      mode_values_stride, mode_indices_stride, tmp_values_stride,
                                                      k, sizes[dim], largest, sorted, data, strides, n);
        };
        
        int64_t grain_size = GRAIN_SIZE / std::max(int64_t{1}, sizes[dim]);
        iter.for_each(loop, /*grain_size=*/grain_size);
    });
}

REGISTER_DISPATCH(sort_stub, &sort_kernel);
REGISTER_DISPATCH(topk_stub, &topk_kernel);

}   // end namespace otter
