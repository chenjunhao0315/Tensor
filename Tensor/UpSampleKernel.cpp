//
//  UpSampleKernel.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/3/3.
//

#include "UpSample.hpp"
#include "UpSampleKernel.hpp"
#include "Dispatch.hpp"
#include "Tensor.hpp"
#include "TensorIterator.hpp"
#include "TensorFactory.hpp"
#include "Parallel.hpp"
#include "Vec.hpp"

namespace otter {

using scale_t = std::vector<double>;

template<int m>
static inline bool is_zero_stride(const int64_t* strides) {
    bool output = strides[0] == 0;
    for (const auto i : otter::irange(1, m)) {
        output &= (strides[i] == 0);
    }
    return output;
}

template <typename scalar_t, typename index_t, int interp_size>
static inline bool is_contiguous_stride(const int64_t* strides) {
    bool output = (strides[0] == sizeof(index_t)) && (strides[1] == sizeof(scalar_t));
    for (int i = 2; i < 2 * interp_size; i += 2) {
        output &= (strides[i] == sizeof(index_t)) && (strides[i + 1] == sizeof(scalar_t));
    }
    return output;
}

static inline int64_t nearest_idx(
    int64_t output_index,
    int64_t input_size,
    int64_t output_size,
    double scales) {
    // This method specificly treats cases: output_size == input_size or
    // output_size == 2 * input_size, that we would like to get rid of
    // We keep this method for BC and consider as deprecated.
    // See nearest_exact_idx as replacement
     if (output_size == input_size) {
         // scale_factor = 1, simply copy
         return output_index;
    } else if (output_size == 2 * input_size) {
        // scale_factor = 2, shift input index
        return output_index >> 1;
    } else {
        float scale = compute_scales_value<float>(scales, input_size, output_size);
        return nearest_neighbor_compute_source_index(scale, output_index, input_size);
    }
}

//static inline int64_t nearest_exact_idx(
//    int64_t output_index,
//    int64_t input_size,
//    int64_t output_size,
//    double scales) {
//    float scale = compute_scales_value<float>(scales, input_size, output_size);
//    return nearest_neighbor_exact_compute_source_index(scale, output_index, input_size);
//}

typedef int64_t (*nearest_idx_fn_t)(int64_t, int64_t, int64_t, double);

template <int n, typename scalar_t, typename index_t, int interp_size>
struct Interpolate {
    static inline scalar_t eval(char* src, char** data, const int64_t* strides, int64_t i) {
        index_t ids = *(index_t*)&data[0][i * strides[0]];
        scalar_t wts = *(scalar_t*)&data[1][i * strides[1]];
        scalar_t t = Interpolate<n - 1, scalar_t, index_t, interp_size>::eval(src + ids, &data[2 * interp_size], &strides[2 * interp_size], i);
        scalar_t output = t * wts;
        for (const auto j : otter::irange(1, interp_size)) {
            ids = *(index_t*)&data[2 * j + 0][i * strides[2 * j + 0]];
            wts = *(scalar_t*)&data[2 * j + 1][i * strides[2 * j + 1]];
            t = Interpolate<n - 1, scalar_t, index_t, interp_size>::eval(src + ids, &data[2 * interp_size], &strides[2 * interp_size], i);
            output += t * wts;
        }
        return output;
    }
};

template <typename scalar_t, typename index_t, int interp_size>
struct Interpolate<1, scalar_t, index_t, interp_size> {
    static inline scalar_t eval(char* src, char** data, const int64_t* strides, int64_t i) {
        index_t ids = *(index_t*)&data[0][i * strides[0]];
        scalar_t wts = *(scalar_t*)&data[1][i * strides[1]];
        scalar_t t = *(scalar_t *)&src[ids];
        scalar_t output = t * wts;
        for (const auto j : otter::irange(1, interp_size)) {
            ids = *(index_t*)&data[2 * j + 0][i * strides[2 * j + 0]];
            wts = *(scalar_t*)&data[2 * j + 1][i * strides[2 * j + 1]];
            t = *(scalar_t *)&src[ids];
            output += t * wts;
        }
        return output;
    }
};

template <int n, typename scalar_t, typename index_t>
struct Interpolate<n, scalar_t, index_t, 1> {
    static inline scalar_t eval(char* src, char** data, const int64_t* strides, int64_t i) {
        index_t ids = *(index_t*)&data[0][i * strides[0]];
        return Interpolate<n - 1, scalar_t, index_t, 1>::eval(src + ids, &data[2], &strides[2], i);
    }
};

template <typename scalar_t, typename index_t>
struct Interpolate<1, scalar_t, index_t, 1> {
    static inline scalar_t eval(char* src, char** data, const int64_t* strides, int64_t i) {
        index_t ids = *(index_t*)&data[0][i * strides[0]];
        return *(scalar_t *)&src[ids];
    }
};

// There is an unexpected 2x slowdown for upsample_trilinear3d channels_first
// for both 1 and 6 threads. We have to specialize this case as below:
// Once the issue is fixed we can keep generic implementation and remove:
// struct Interpolate<n, scalar_t, index_t, 2> and
// struct Interpolate<1, scalar_t, index_t, 2>
template <int n, typename scalar_t, typename index_t>
struct Interpolate<n, scalar_t, index_t, 2> {
    static inline scalar_t eval(char* src, char** data, const int64_t* strides, int64_t i) {
        index_t i0 = *(index_t*)&data[0][i * strides[0]];
        index_t i1 = *(index_t*)&data[2][i * strides[2]];
        scalar_t w0 = *(scalar_t *)&data[1][i * strides[1]];
        scalar_t w1 = *(scalar_t *)&data[3][i * strides[3]];

        scalar_t t0 = Interpolate<n - 1, scalar_t, index_t, 2>::eval(src + i0, &data[4], &strides[4], i);
        scalar_t t1 = Interpolate<n - 1, scalar_t, index_t, 2>::eval(src + i1, &data[4], &strides[4], i);

        return t0 * w0 + t1 * w1;
    }
};

template <typename scalar_t, typename index_t>
struct Interpolate<1, scalar_t, index_t, 2> {
    static inline scalar_t eval(char* src, char** data, const int64_t* strides, int64_t i) {
        index_t i0 = *(index_t*)&data[0][i * strides[0]];
        index_t i1 = *(index_t*)&data[2][i * strides[2]];
        scalar_t w0 = *(scalar_t *)&data[1][i * strides[1]];
        scalar_t w1 = *(scalar_t *)&data[3][i * strides[3]];
        scalar_t t0 = *(scalar_t *)&src[i0];
        scalar_t t1 = *(scalar_t *)&src[i1];
        return t0 * w0 + t1 * w1;
    }
};

template <int n, typename scalar_t, typename index_t, int interp_size>
static inline scalar_t interpolate(char* src, char** data, const int64_t* strides, int64_t i) {
    return Interpolate<n, scalar_t, index_t, interp_size>::eval(src, data, strides, i);
}

template <int N, int non_zero_stride_dim, typename scalar_t, typename index_t, int interp_size>
struct CheckAlmostAllZeroStrides {
    static inline bool eval(const int64_t* strides) {
    // N is dim index: N -> dim0, N-1 -> dim1, ...
    // non_zero_stride_dim should be out_dims - dim
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    bool output;
    if (N == non_zero_stride_dim) {
        output = is_contiguous_stride<scalar_t, index_t, interp_size>(strides);
    } else {
        output = is_zero_stride<2 * interp_size>(strides);
    }
    return output &&
        CheckAlmostAllZeroStrides<N - 1, non_zero_stride_dim, scalar_t, index_t, interp_size>::eval(
                                                                                                    &strides[2 * interp_size]);
    }
};

template <int non_zero_stride_dim, typename scalar_t, typename index_t, int interp_size>
struct CheckAlmostAllZeroStrides<0, non_zero_stride_dim, scalar_t, index_t, interp_size> {
    static inline bool eval(const int64_t* strides) {
        return true;
    }
};

template <int n, int s, typename scalar_t, typename index_t, int interp_size>
static inline bool check_almost_all_zero_stride(const int64_t* strides) {
    return CheckAlmostAllZeroStrides<n, s, scalar_t, index_t, interp_size>::eval(strides);
}

template <typename scalar_t, typename index_t, int out_ndims, int interp_size>
static inline void basic_loop(char** data, const int64_t* strides, int64_t n) {
    char* dst = data[0];
    char* src = data[1];
    for (const auto i : otter::irange(n)) {
        *(scalar_t*)&dst[i * strides[0]] = interpolate<out_ndims, scalar_t, index_t, interp_size>(
            src + i * strides[1], &data[2], &strides[2], i);
    }
}

template <typename scalar_t, int out_ndims, int interp_size>
void cpu_upsample_generic(TensorIterator& iter) {
    auto loop = [&](char** data, const int64_t* strides, int64_t n) {
        // special-cases to let the compiler apply compile-time input-specific optimizations
        if ((strides[0] == sizeof(scalar_t) && (strides[1] == 0) && check_almost_all_zero_stride<out_ndims, 1, scalar_t, int64_t, interp_size>(&strides[2]))) {
            // contiguous channels-first case
            basic_loop<scalar_t, int64_t, out_ndims, interp_size>(data, strides, n);
        } else if ((strides[0] == sizeof(scalar_t) && (strides[1] == sizeof(scalar_t)) && check_almost_all_zero_stride<out_ndims, -1, scalar_t, int64_t, interp_size>(&strides[2]))) {
            // contiguous channels-last case
            basic_loop<scalar_t, int64_t, out_ndims, interp_size>(data, strides, n);
        } else {
            // fallback
            basic_loop<scalar_t, int64_t, out_ndims, interp_size>(data, strides, n);
        }
    };
    iter.for_each(loop);
}

template <typename scalar_t, typename scale_type, nearest_idx_fn_t nearest_idx_fn>
void cpu_upsample_nearest_channels_last(
    const Tensor& output_,
    const Tensor& input_,
    const scale_type& scales) {
//    OTTER_CHECK(input_.dtype() == output_.dtype(), "expected dtype ", input_.dtype(), " for `output` but got dtype ", output_.dtype());

    auto input_sizes = input_.sizes().vec();
    auto output_sizes = output_.sizes().vec();
    auto ndim = input_sizes.size();
    OTTER_CHECK(ndim >=4 && ndim <= 5, "Upsample with NHWC format supports tensors with 4 or 5 dims.")

    auto channels_last_memory_format = ndim == 4 ? MemoryFormat::ChannelsLast : MemoryFormat::ChannelsLast3d;
    auto input = input_.contiguous(channels_last_memory_format);
    auto output = output_.contiguous(channels_last_memory_format);

    auto input_data = input.data_ptr<scalar_t>();
    auto output_data = output.data_ptr<scalar_t>();

    int64_t num_batches =  input_sizes[0];
    int64_t channels =  input_sizes[1];
    int64_t input_depth = (ndim == 5) ? input_sizes[2] : 1;
    int64_t output_depth = (ndim == 5) ? output_sizes[2] : 1;
    int64_t input_height = (ndim >= 4) ? input_sizes[ndim - 2] : 1;
    int64_t output_height = (ndim >= 4) ? output_sizes[ndim - 2] : 1;
    int64_t input_width = input_sizes[ndim - 1];
    int64_t output_width = output_sizes[ndim - 1];
    int64_t numel = output.numel();

    OTTER_CHECK(channels > 0, "expected input and output channels greater than 0 but got ", channels);

    using Vec = vec::Vectorized<scalar_t>;
    auto copy = [](scalar_t* out, scalar_t* in, int64_t size) {
        int64_t d = 0;
        for (; d < size - (size % Vec::size()); d += Vec::size()) {
            Vec out_vec = Vec::loadu(in + d);
            out_vec.store(out + d);
        }
        for (; d < size; d++) {
            out[d] = in[d];
        }
    };

    auto loop2d = [&](int64_t begin, int64_t end) {
        int64_t n = 0;
        int64_t oh = 0;
        int64_t ow = 0;
        data_index_init(begin, n, num_batches, oh, output_height, ow, output_width);

        for (const auto i : otter::irange(begin, end)) {
            int64_t ih = nearest_idx_fn(oh, input_height, output_height, scales[0]);
            int64_t iw = nearest_idx_fn(ow, input_width, output_width, scales[1]);
            scalar_t* output_ptr = output_data + i * channels;
            scalar_t* input_ptr = input_data + n * input_height * input_width * channels + ih * input_width * channels + iw * channels;
            copy(output_ptr, input_ptr, channels);
            data_index_step(n, num_batches, oh, output_height, ow, output_width);
        }
    };

    auto loop3d = [&](int64_t begin, int64_t end) {
        int64_t n = 0;
        int64_t od = 0;
        int64_t oh = 0;
        int64_t ow = 0;
        data_index_init(begin, n, num_batches, od, output_depth, oh, output_height, ow, output_width);

        for (const auto i : otter::irange(begin, end)) {
            int64_t id = nearest_idx_fn(od, input_depth, output_depth, scales[0]);
            int64_t ih = nearest_idx_fn(oh, input_height, output_height, scales[1]);
            int64_t iw = nearest_idx_fn(ow, input_width, output_width, scales[2]);
            scalar_t* output_ptr = output_data + i * channels;
            scalar_t* input_ptr = input_data + n * input_depth * input_height * input_width * channels + id * input_height * input_width * channels + ih * input_width * channels + iw * channels;
            copy(output_ptr, input_ptr, channels);
            data_index_step(n, num_batches, od, output_depth, oh, output_height, ow, output_width);
        }
    };

    if (ndim == 4) {
        // upsample nearest 2d
        otter::parallel_for(0, numel / channels, otter::GRAIN_SIZE / channels, loop2d);
    } else {
        // upsample nearest 3d
        OTTER_INTERNAL_ASSERT(ndim == 5);
        otter::parallel_for(0, numel / channels, otter::GRAIN_SIZE / channels, loop3d);
    }

    if (!output_.is_contiguous(channels_last_memory_format)) {
        output_.copy_(output);
    }
}

template <int out_ndims, typename scale_type, class F>
void upsample_generic_Nd_kernel_impl(
    const Tensor& output,
    const Tensor& input,
    bool align_corners,
    const scale_type& scales) {
    
    // input can be NCHW, NCL or NCKHW
    auto shape = input.sizes().vec();
    auto strides = input.strides().vec();
    auto oshape = output.sizes();

    OTTER_INTERNAL_ASSERT(shape.size() == oshape.size() && shape.size() == 2 + out_ndims);
    OTTER_INTERNAL_ASSERT(strides.size() == 2 + out_ndims);

    for (const auto i : otter::irange(out_ndims)) {
        shape[i + 2] = oshape[i + 2];
        strides[i + 2] = 0;
    }
    auto restrided_input = input.as_strided(shape, strides);
    
    std::vector<std::vector<Tensor>> indices_weights;

    constexpr int interp_size = F::interp_size;
    auto input_scalar_type = input.scalar_type();
    if (interp_size == 1 && input_scalar_type == ScalarType::Byte) {
        // with compute_indices_weights
        input_scalar_type = ScalarType::Float;
    }

    for (const auto i : otter::irange(out_ndims)) {
        indices_weights.emplace_back(
            F::compute_indices_weights(
                input_scalar_type, input.size(i + 2), oshape[i + 2],
                input.stride(i + 2) * input.itemsize(),
                input.dim(), i + 2, align_corners, scales[i]
            )
        );
    }

    TensorIteratorConfig config;
    config.check_all_same_dtype(false)
        .declare_static_dtype_and_device(input.scalar_type(), input.device())
        .add_output(output)
        .add_input(restrided_input);

    for (auto & idx_weight: indices_weights) {
        for (auto& tensor : idx_weight) {
            config.add_input(tensor);
        }
    }

    auto iter = config.build();
    
    if (interp_size > 1) {
        // Nearest also supports uint8 tensor, so need to handle it separately
        OTTER_DISPATCH_FLOATING_TYPES(iter.dtype(), "upsample_generic_Nd", [&] {
            // MSVC can not catch constexpr int interp_size here
            constexpr int mode = F::interp_size;
            cpu_upsample_generic<scalar_t, out_ndims, mode>(iter);
        });
    } else {
        OTTER_DISPATCH_FLOATING_TYPES_AND(ScalarType::Byte, iter.dtype(), "upsample_generic_Nd", [&] {
            constexpr int mode = F::interp_size;
            cpu_upsample_generic<scalar_t, out_ndims, mode>(iter);
        });
    }
}

struct HelperInterpBase {
    static inline void init_indices_weights(
        ScalarType output_type,
        std::vector<Tensor> & output, int64_t output_size, int64_t ndims,
        int64_t reshape_dim, int interp_size
    ) {

        auto new_shape = std::vector<int64_t>(ndims, 1);
        new_shape[reshape_dim] = output_size;

        for (const auto j : otter::irange(interp_size)) {
            (void)j; //Suppress unused variable warning
            output.emplace_back(otter::empty(new_shape, otter::CppTypeToScalarType<int64_t>()));
            output.emplace_back(otter::empty(new_shape, output_type));
        }
    }

    template <typename scalar_t, typename aa_filter_fn_t>
    static inline void _compute_weights_aa(
        const int64_t i, const int64_t input_size, const scalar_t scale, const scalar_t support,
        scalar_t* wt_ptr, const int64_t interp_size, aa_filter_fn_t filter_fn,
        int64_t& xmin, int64_t& xsize
    ) {

        scalar_t center = scale * (i + 0.5);
        scalar_t total_w = 0.0;
        scalar_t invscale = (scale >= 1.0) ? 1.0 / scale : 1.0;
        xmin = std::max(static_cast<int64_t>(center - support + 0.5), static_cast<int64_t>(0));
        xsize = std::min(static_cast<int64_t>(center + support + 0.5), input_size) - xmin;

        int64_t j = 0;
        for (; j < xsize; j++) {
            scalar_t w = filter_fn((j + xmin - center + 0.5) * invscale);
            wt_ptr[j] = w;
            total_w += w;
        }
        for (j = 0; j < xsize; j++) {
            if (total_w != 0.0) {
                wt_ptr[j] /= total_w;
            }
        }
        for (; j < interp_size; j++) {
            wt_ptr[j] = static_cast<scalar_t>(0.0);
        }
    }

    template <typename scalar_t, typename aa_filter_fn_t>
    static inline std::vector<Tensor> _compute_indices_weights_aa(
        int64_t input_size, int64_t output_size, int64_t stride, int64_t ndims,
        int64_t reshape_dim, bool align_corners, scalar_t scale,
        int interp_size, aa_filter_fn_t aa_filter_fn
    ) {

        std::vector<Tensor> output;

        scalar_t support = (scale >= 1.0) ? (interp_size * 0.5) * scale : interp_size * 0.5;
        interp_size = (int)ceilf(support) * 2 + 1;

        auto new_shape = std::vector<int64_t>(ndims, 1);
        new_shape[reshape_dim] = output_size;

        // Bounds approach as in PIL: xmin/xmax
        output.emplace_back(otter::empty(new_shape, otter::CppTypeToScalarType<int64_t>()));
        output.emplace_back(otter::empty(new_shape, otter::CppTypeToScalarType<int64_t>()));
        output.emplace_back(otter::empty(new_shape, otter::CppTypeToScalarType<int64_t>()));

        {
            // Weights
            new_shape[reshape_dim] = output_size * interp_size;
            auto wts = empty(new_shape, otter::CppTypeToScalarType<scalar_t>());
            auto strides = wts.strides().vec();
            strides[reshape_dim] = 0;
            new_shape[reshape_dim] = output_size;
            wts = wts.as_strided(new_shape, strides);
            output.emplace_back(wts);
            // Weights indices
            output.emplace_back(otter::empty(new_shape, otter::CppTypeToScalarType<int64_t>()));
        }

        int64_t* idx_ptr_xmin = output[0].data_ptr<int64_t>();
        int64_t* idx_ptr_size = output[1].data_ptr<int64_t>();
        int64_t* idx_ptr_stride = output[2].data_ptr<int64_t>();
        scalar_t* wt_ptr = output[3].data_ptr<scalar_t>();
        int64_t* wt_idx_ptr = output[4].data_ptr<int64_t>();
        int64_t xmin, xmax;
        for (const auto i : otter::irange(output_size)) {
            HelperInterpBase::_compute_weights_aa(
                i,
                input_size,
                scale,
                support,
                wt_ptr + i * interp_size,
                interp_size,
                aa_filter_fn,
                xmin,
                xmax);
            idx_ptr_xmin[i] = xmin * stride;
            idx_ptr_size[i] = xmax;
            idx_ptr_stride[i] = stride;
            wt_idx_ptr[i] = i * interp_size * sizeof(scalar_t);
        }
        return output;
    }
};

struct HelperInterpNearest : public HelperInterpBase {
    // This structure implements outdated and buggy method to compute indices
    // for nearest neighbours interpolation
    // We keep this structure for BC and consider as deprecated.
    // See HelperInterpNearestExact as replacement

    static const int interp_size = 1;

    static inline void init_indices_weights(
                                            ScalarType output_type,
                                            std::vector<Tensor> & output, int64_t output_size, int64_t ndims,
                                            int64_t reshape_dim, int interp_size
    ) {
        auto new_shape = std::vector<int64_t>(ndims, 1);
        new_shape[reshape_dim] = output_size;

        for (const auto j : otter::irange(interp_size)) {
            (void)j; //Suppress unused variable warning
            output.emplace_back(otter::empty(new_shape, otter::CppTypeToScalarType<int64_t>()));
            // Defines weights for consistency, but not used
            output.emplace_back(otter::ones(new_shape, output_type));
        }
    }

    // Compute nearest mode indices and weights for each interpolated dimension
    // indices_weights = {
    //      {indices_0, 1.0, },  // dim -n
    //      {indices_0, 1.0, },  // dim -(n-1)
    //      ...
    //      {indices_0, 1.0, },  // dim -1
    // }
    // Indices and weights are reshaped as (1, 1, ..., N, ..., 1, 1) to
    // fit input/output tensors.
    // Indices are already containing the strides to optimize the computations
    static inline std::vector<Tensor> compute_indices_weights(
        ScalarType scalar_type,
        int64_t input_size, int64_t output_size, int64_t stride, int64_t ndims,
        int64_t reshape_dim, bool align_corners, const double opt_scale
    ) {

        OTTER_INTERNAL_ASSERT(!align_corners);
        // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
        std::vector<Tensor> output;
        HelperInterpNearest::init_indices_weights(scalar_type, output, output_size, ndims, reshape_dim, HelperInterpNearest::interp_size);

        OTTER_DISPATCH_FLOATING_TYPES(scalar_type, "compute_indices_weights_nearest", [&] {

            scalar_t scale = area_pixel_compute_scale<scalar_t>(input_size, output_size, align_corners, opt_scale);

            auto input_index_ptr = output[0].data_ptr<int64_t>();
            int64_t input_index;

            // Indices are computed as following:
            // scale = 1.0 * isize / osize
            // index_f32 = (output_index) * scale
            // input_index = floor(index_f32)
            // Same as OpenCV INTER_N
            for (const auto i : otter::irange(output_size)) {
                const scalar_t real_input_index = area_pixel_compute_source_index<scalar_t>(scale, i, /*align_corners=*/true, /*cubic=*/false);
                input_index = static_cast<int64_t>(floorf(real_input_index));
                input_index_ptr[i] = static_cast<int64_t>(std::min(input_index, input_size - 1)) * stride;
            }
        });
        return output;
    }

};

void upsample_nearest2d_kernel_impl(
    const Tensor& output,
    const Tensor& input,
    double scales_h,
    double scales_w) {
    
    bool check;
    
    if ((check = input.is_contiguous(MemoryFormat::ChannelsLast))) {
        OTTER_DISPATCH_FLOATING_TYPES(input.scalar_type(), "upsample_nearest2d_channels_last", [&] {
            cpu_upsample_nearest_channels_last<scalar_t, scale_t, nearest_idx>(output, input, {scales_h, scales_w});
        });
    } else {
        upsample_generic_Nd_kernel_impl<2, scale_t, HelperInterpNearest>(output, input, false, {scales_h, scales_w});
    }
}

REGISTER_DISPATCH(upsampling_nearest2d_stub, &upsample_nearest2d_kernel_impl);

}   // end namespace otter
