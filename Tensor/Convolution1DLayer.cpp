//
//  Convolution1DLayer.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/7/11.
//

#include "Convolution1DLayer.hpp"

#include "Tensor.hpp"
#include "TensorFactory.hpp"
#include "TensorMaker.hpp"
#include "Parallel.hpp"
#include "Padding.hpp"

#include "QuantizeX86.hpp"
#include "ActivationLayer.hpp"
#include "TensorPacking.hpp"

namespace otter {

Convolution1DLayer::Convolution1DLayer() {
    one_blob_only = true;
    support_inplace = false;
#if __SSE2__
    support_packing = true;
#endif // __SSE2__
}

Convolution1DLayer::~Convolution1DLayer() {
    if (activation) {
        delete activation;
        activation = nullptr;
    }
}

int Convolution1DLayer::parse_param(LayerOption& option, ParamDict& pd) {
    pd.clear();
    int in_channels   = opt_find_int(option, "in_channels", 1);
    int out_channels  = opt_find_int(option, "out_channels", 1);
    int kernel_w  = opt_find_int(option, "kernel_w", -1);
    int kernel        = opt_find_int(option, "kernel", 3);
    if (kernel_w < 1) {
        if (kernel_w < 0)  kernel_w  = kernel;
    }
    int stride_w  = opt_find_int(option, "stride_w", -1);
    int stride        = opt_find_int(option, "stride", 1);
    if (stride_w < 1) {
        if (stride_w < 0)  stride_w  = stride;
    }
    int padding_w  = opt_find_int(option, "padding_w", -1);
    int padding        = opt_find_int(option, "padding", 0);
    if (padding_w < 0) {
        if (padding_w < 0)  padding_w  = padding;
    }
    int dilation_w  = opt_find_int(option, "dilation_w", -1);
    int dilation        = opt_find_int(option, "dilation", 1);
    if (dilation_w < 1) {
        if (dilation_w < 0)  dilation_w  = dilation;
    }
    if (opt_find(option, "bias_term")) {
        if (option["bias_term"] == "false")
            bias_term = 0;
        else
            bias_term = 1;
    }
    
    std::string activation = opt_find_string(option, "activation", "");
    
    int activation_type = 0;
    if (activation == "Relu") {
        activation_type = 1;
    } else if (activation == "LRelu") {
        activation_type = 2;
    } else if (activation == "Relu6") {
        activation_type = 3;
    } else if (activation == "Sigmoid") {
        activation_type = 4;
    }
    
    Tensor activation_params;
    if (opt_check_string(option, "activation_params")) {
        int num_params = (int)std::count(option["activation_params"].begin(), option["activation_params"].end(), ',') + 1;
        activation_params = otter::empty({num_params}, otter::ScalarType::Float);
        auto activation_params_a = activation_params.accessor<float, 1>();
        std::stringstream ss;
        ss << option["activation_params"];
        float n; char c;
        for (const auto i : otter::irange(num_params)) {
            ss >> n >> c;
            activation_params_a[i] = n;
        }
    }
    
    pd.set((int)Conv1DParam::In_channels, in_channels);
    pd.set((int)Conv1DParam::Out_channels, out_channels);
    pd.set((int)Conv1DParam::Kernel_width, kernel_w);
    pd.set((int)Conv1DParam::Stride_width,  stride_w);
    pd.set((int)Conv1DParam::Padding_width,  padding_w);
    pd.set((int)Conv1DParam::Dilation_width,  dilation_w);
    pd.set((int)Conv1DParam::Bias_term, bias_term);
    pd.set((int)Conv1DParam::Activation_type, activation_type);
    pd.set((int)Conv1DParam::Activation_params, activation_params);
    pd.set((int)Conv1DParam::Groups, 1);
    
    return 0;
}

int Convolution1DLayer::compute_output_shape(ParamDict& pd) {
    auto shape_a = bottom_shapes[0].accessor<int, 2>()[0];
    int input_batch    = shape_a[0];
    int input_channels = shape_a[1];
    int input_length   = shape_a[2];
    int out_channels    = pd.get((int)Conv1DParam::Out_channels, 1);
    int kernel_w    = pd.get((int)Conv1DParam::Kernel_width, 3);
    int stride_w    = pd.get((int)Conv1DParam::Stride_width,  1);
    int padding_w   = pd.get((int)Conv1DParam::Padding_width,  0);
    int dilation_w  = pd.get((int)Conv1DParam::Dilation_width,  1);
    int groups          = pd.get((int)Conv1DParam::Groups, 1);
    int out_length  = (input_length + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;
    
    pd.set((int)Conv1DParam::In_channels, input_channels);
    pd.set(OUTPUT_SHAPE_HINT, otter::tensor({input_batch, out_channels, out_length}, ScalarType::Int).view({1, -1}));
    
    return 0;
}

int Convolution1DLayer::load_param(const ParamDict &pd) {
    in_channels     = pd.get((int)Conv1DParam::In_channels, 1);
    out_channels    = pd.get((int)Conv1DParam::Out_channels, 1);
    kernel_w    = pd.get((int)Conv1DParam::Kernel_width, 3);
    stride_w    = pd.get((int)Conv1DParam::Stride_width,  1);
    padding_w   = pd.get((int)Conv1DParam::Padding_width,  0);
    dilation_w  = pd.get((int)Conv1DParam::Dilation_width,  1);
    bias_term = pd.get((int)Conv1DParam::Bias_term, 0);
    activation_type = pd.get((int)Conv1DParam::Activation_type, 0);
    activation_params = pd.get((int)Conv1DParam::Activation_params, Tensor());
    
    return 0;
}

int Convolution1DLayer::init_model() {
    
    weight_data = otter::rand({out_channels, in_channels / groups, kernel_w}, ScalarType::Float);
    if (bias_term)
        bias_data = otter::rand({out_channels}, ScalarType::Float);
    
    return 0;
}

int Convolution1DLayer::load_model(const Initializer& initializer) {
    weight_data = initializer.load({out_channels, in_channels / groups, kernel_w}, 0);
    
    if (bias_term) {
        bias_data = initializer.load({out_channels}, 1);
    }
    
    return 0;
}

int Convolution1DLayer::create_pipeline(const NetOption& opt) {
    
    activation = create_activation_layer(activation_type, activation_params);
    
    int elempack = 1;
    int out_elempack = 1;

#if __SSE2__
    if (opt.use_packing_layout)
    {
#if __AVX__
        elempack = in_channels % 8 == 0 ? 8 : in_channels % 4 == 0 ? 4 : 1;
        out_elempack = out_channels % 8 == 0 ? 8 : out_channels % 4 == 0 ? 4 : 1;
#else
        elempack = in_channels % 4 == 0 ? 4 : 1;
        out_elempack = out_channels % 4 == 0 ? 4 : 1;
#endif
    }
#endif // __SSE2__

    // src = kw-inch-outch
    // dst = pb-pa-kw-inch/pa-outch/pb
    {
        Tensor weight_data_r2 = weight_data.view({out_channels, in_channels, kernel_w});
        
        weight_data_packed = otter::empty({out_channels / out_elempack, in_channels / elempack, kernel_w}, otter::get_update_scalarType(otter::ScalarType::Float, elempack * out_elempack));
        
        auto weight_data_r2_a = weight_data_r2.raw_accessor<float, 3>();
        auto weight_data_packed_ra = weight_data_packed.raw_accessor<float, 3>();

        for (int q = 0; q + (out_elempack - 1) < out_channels; q += out_elempack)
        {
            float* g00 = weight_data_packed_ra[q / out_elempack].data();

            for (int p = 0; p + (elempack - 1) < in_channels; p += elempack)
            {
                for (int k = 0; k < kernel_w; k++)
                {
                    for (int i = 0; i < elempack; i++)
                    {
                        for (int j = 0; j < out_elempack; j++)
                        {
                            const float* k00 = weight_data_r2_a[q + j][p + i].data();

                            g00[0] = k00[k];

                            g00++;
                        }
                    }
                }
            }
        }
    }

    return 0;
}

int Convolution1DLayer::forward(const Tensor& bottom_blob, Tensor& top_blob, const NetOption& opt) const {
    int w = bottom_blob.size(1);
    int h = bottom_blob.size(0);
    int elempack = bottom_blob.elempack();

    const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;

    Tensor bottom_blob_bordered = otter::constant_pad(bottom_blob, {padding_w, padding_w
        
    }, 0);

    w = bottom_blob_bordered.size(1);
    h = bottom_blob_bordered.size(0);

    int out_elempack = 1;
#if __SSE2__
    if (opt.use_packing_layout)
    {
#if __AVX__
        out_elempack = out_channels % 8 == 0 ? 8 : out_channels % 4 == 0 ? 4 : 1;
#else
        out_elempack = out_channels % 4 == 0 ? 4 : 1;
#endif
    }
#endif // __SSE2__

    const int outw = (w - kernel_extent_w) / stride_w + 1;
    const int outh = out_channels / out_elempack;
    
    top_blob = otter::empty({outh, outw}, otter::get_update_scalarType(otter::ScalarType::Float, out_elempack));
    
    auto bottom_blob_bordered_ra = bottom_blob_bordered.raw_accessor<float, 2>();
    auto top_blob_ra = top_blob.raw_accessor<float, 2>();
    const float* bias_data_ptr = bias_term ? bias_data.data_ptr<float>() : nullptr;

#if __SSE2__
#if __AVX__
    if (elempack == 8 && out_elempack == 8)
    {
        auto weight_data_packed_ra = weight_data_packed.raw_accessor<float, 3>();
        {
            otter::parallel_for(0, outh, 0, [&](int64_t begin, int64_t end) {
                for (const auto p : otter::irange(begin, end))
                {
                    float* outptr = top_blob_ra[p].data();

                    for (int j = 0; j < outw; j++)
                    {
                        __m256 _sum = _mm256_set1_ps(0.f);

                        if (bias_term)
                        {
                            _sum = _mm256_loadu_ps(((const float*)bias_data_ptr) + p * 8);
                        }

                        const float* kptr = weight_data_packed_ra[p].data();

                        for (int q = 0; q < h; q++)
                        {
                            const float* sptr = bottom_blob_bordered_ra[q].data() + j * stride_w * 8;

                            for (int k = 0; k < kernel_w; k++)
                            {
                                __m256 _val0 = _mm256_broadcast_ss(sptr);
                                __m256 _val1 = _mm256_broadcast_ss(sptr + 1);
                                __m256 _val2 = _mm256_broadcast_ss(sptr + 2);
                                __m256 _val3 = _mm256_broadcast_ss(sptr + 3);
                                __m256 _val4 = _mm256_broadcast_ss(sptr + 4);
                                __m256 _val5 = _mm256_broadcast_ss(sptr + 5);
                                __m256 _val6 = _mm256_broadcast_ss(sptr + 6);
                                __m256 _val7 = _mm256_broadcast_ss(sptr + 7);

                                __m256 _w0 = _mm256_loadu_ps(kptr);
                                __m256 _w1 = _mm256_loadu_ps(kptr + 8);
                                __m256 _w2 = _mm256_loadu_ps(kptr + 16);
                                __m256 _w3 = _mm256_loadu_ps(kptr + 24);
                                __m256 _w4 = _mm256_loadu_ps(kptr + 32);
                                __m256 _w5 = _mm256_loadu_ps(kptr + 40);
                                __m256 _w6 = _mm256_loadu_ps(kptr + 48);
                                __m256 _w7 = _mm256_loadu_ps(kptr + 56);

                                _mm256_comp_fmadd_ps8(_sum,
                                                      _val0, _val1, _val2, _val3, _val4, _val5, _val6, _val7,
                                                      _w0, _w1, _w2, _w3, _w4, _w5, _w6, _w7);

                                sptr += dilation_w * 8;
                                kptr += 64;
                            }
                        }

                        _sum = activation_avx(_sum, activation_type, activation_params);

                        _mm256_storeu_ps(outptr, _sum);
                        outptr += 8;
                    }
                }
            });
        }
    }

    if (elempack == 1 && out_elempack == 8)
    {
        auto weight_data_packed_ra = weight_data_packed.raw_accessor<float, 3>();
        {
            otter::parallel_for(0, outh, 0, [&](int64_t begin, int64_t end) {
                for (const auto p : otter::irange(begin, end)) {
                    float* outptr = top_blob_ra[p].data();

                    for (int j = 0; j < outw; j++)
                    {
                        __m256 _sum = _mm256_set1_ps(0.f);

                        if (bias_term)
                        {
                            _sum = _mm256_loadu_ps(((const float*)bias_data_ptr) + p * 8);
                        }

                        const float* kptr = weight_data_packed_ra[p].data();

                        for (int q = 0; q < h; q++)
                        {
                            const float* sptr = bottom_blob_bordered_ra[q].data() + j * stride_w;

                            for (int k = 0; k < kernel_w; k++)
                            {
                                __m256 _val = _mm256_set1_ps(sptr[0]);
                                __m256 _w = _mm256_loadu_ps(kptr);
                                _sum = _mm256_comp_fmadd_ps(_val, _w, _sum);

                                sptr += dilation_w;
                                kptr += 8;
                            }
                        }

                        _sum = activation_avx(_sum, activation_type, activation_params);

                        _mm256_storeu_ps(outptr, _sum);
                        outptr += 8;
                    }
                }
            });
        }
    }

    if (elempack == 4 && out_elempack == 8)
    {
        auto weight_data_packed_ra = weight_data_packed.raw_accessor<float, 3>();
        {
            otter::parallel_for(0, outh, 0, [&](int64_t begin, int64_t end) {
                for (const auto p : otter::irange(begin, end))
                {
                    float* outptr = top_blob_ra[p].data();

                    for (int j = 0; j < outw; j++)
                    {
                        __m256 _sum = _mm256_set1_ps(0.f);

                        if (bias_term)
                        {
                            _sum = _mm256_loadu_ps((const float*)bias_data_ptr + p * 8);
                        }

                        const float* kptr = weight_data_packed_ra[p].data();

                        for (int q = 0; q < h; q++)
                        {
                            const float* sptr = bottom_blob_bordered_ra[q].data() + j * stride_w * 4;

                            for (int k = 0; k < kernel_w; k++)
                            {
                                __m256 _val0 = _mm256_broadcast_ss(sptr);
                                __m256 _val1 = _mm256_broadcast_ss(sptr + 1);
                                __m256 _val2 = _mm256_broadcast_ss(sptr + 2);
                                __m256 _val3 = _mm256_broadcast_ss(sptr + 3);

                                __m256 _w0 = _mm256_loadu_ps(kptr);
                                _sum = _mm256_comp_fmadd_ps(_val0, _w0, _sum);
                                __m256 _w1 = _mm256_loadu_ps(kptr + 8);
                                _sum = _mm256_comp_fmadd_ps(_val1, _w1, _sum);
                                __m256 _w2 = _mm256_loadu_ps(kptr + 16);
                                _sum = _mm256_comp_fmadd_ps(_val2, _w2, _sum);
                                __m256 _w3 = _mm256_loadu_ps(kptr + 24);
                                _sum = _mm256_comp_fmadd_ps(_val3, _w3, _sum);

                                sptr += dilation_w * 4;
                                kptr += 32;
                            }
                        }

                        _sum = activation_avx(_sum, activation_type, activation_params);

                        _mm256_storeu_ps(outptr, _sum);
                        outptr += 8;
                    }
                }
            });
        }
    }

    if (elempack == 8 && out_elempack == 1)
    {
        auto weight_data_packed_ra = weight_data_packed.raw_accessor<float, 3>();
        {
            otter::parallel_for(0, outh, 0, [&](int64_t begin, int64_t end) {
                for (const auto p : otter::irange(begin, end))
                {
                    float* outptr = top_blob_ra[p].data();

                    for (int j = 0; j < outw; j++)
                    {
                        float sum = 0.f;

                        if (bias_term)
                        {
                            sum = bias_data_ptr[p];
                        }

                        const float* kptr = weight_data_packed_ra[p].data();

                        __m256 _sum8 = _mm256_set1_ps(0);

                        for (int q = 0; q < h; q++)
                        {
                            const float* sptr = bottom_blob_bordered_ra[q].data() + j * stride_w * 8;

                            for (int k = 0; k < kernel_w; k++) // 29.23
                            {
                                __m256 _val = _mm256_loadu_ps(sptr);
                                __m256 _w = _mm256_loadu_ps(kptr);
                                __m256 _s8 = _mm256_mul_ps(_val, _w);
                                _sum8 = _mm256_add_ps(_sum8, _s8);

                                sptr += dilation_w * 8;
                                kptr += 8;
                            }
                        }
                        sum += _mm256_reduce_add_ps(_sum8); // dot
                        sum = activation_ss(sum, activation_type, activation_params);

                        outptr[j] = sum;
                    }
                }
            });
        }
    }

    if (elempack == 8 && out_elempack == 4)
    {
        auto weight_data_packed_ra = weight_data_packed.raw_accessor<float, 3>();
        {
            otter::parallel_for(0, outh, 0, [&](int64_t begin, int64_t end) {
                for (const auto p : otter::irange(begin, end))
                {
                    float* outptr = top_blob_ra[p].data();

                    for (int j = 0; j < outw; j++)
                    {
                        __m128 _sum = _mm_set1_ps(0.f);

                        if (bias_term)
                        {
                            _sum = _mm_loadu_ps((const float*)bias_data_ptr + p * 4);
                        }

                        const float* kptr = weight_data_packed_ra[p].data();

                        for (int q = 0; q < h; q++)
                        {
                            const float* sptr = bottom_blob_bordered_ra[q].data() + j * stride_w * 8;

                            for (int k = 0; k < kernel_w; k++)
                            {
                                __m128 _val0 = _mm_broadcast_ss(sptr);
                                __m128 _val1 = _mm_broadcast_ss(sptr + 1);
                                __m128 _val2 = _mm_broadcast_ss(sptr + 2);
                                __m128 _val3 = _mm_broadcast_ss(sptr + 3);
                                __m128 _val4 = _mm_broadcast_ss(sptr + 4);
                                __m128 _val5 = _mm_broadcast_ss(sptr + 5);
                                __m128 _val6 = _mm_broadcast_ss(sptr + 6);
                                __m128 _val7 = _mm_broadcast_ss(sptr + 7);

                                __m128 _w0 = _mm_loadu_ps(kptr);
                                _sum = _mm_comp_fmadd_ps(_val0, _w0, _sum);
                                __m128 _w1 = _mm_loadu_ps(kptr + 4);
                                _sum = _mm_comp_fmadd_ps(_val1, _w1, _sum);
                                __m128 _w2 = _mm_loadu_ps(kptr + 8);
                                _sum = _mm_comp_fmadd_ps(_val2, _w2, _sum);
                                __m128 _w3 = _mm_loadu_ps(kptr + 12);
                                _sum = _mm_comp_fmadd_ps(_val3, _w3, _sum);
                                __m128 _w4 = _mm_loadu_ps(kptr + 16);
                                _sum = _mm_comp_fmadd_ps(_val4, _w4, _sum);
                                __m128 _w5 = _mm_loadu_ps(kptr + 20);
                                _sum = _mm_comp_fmadd_ps(_val5, _w5, _sum);
                                __m128 _w6 = _mm_loadu_ps(kptr + 24);
                                _sum = _mm_comp_fmadd_ps(_val6, _w6, _sum);
                                __m128 _w7 = _mm_loadu_ps(kptr + 28);
                                _sum = _mm_comp_fmadd_ps(_val7, _w7, _sum);

                                sptr += dilation_w * 8;
                                kptr += 32;
                            }
                        }

                        _sum = activation_sse(_sum, activation_type, activation_params);

                        _mm_storeu_ps(outptr, _sum);
                        outptr += 4;
                    }
                }
            });
        }
    }
#endif

    if (elempack == 4 && out_elempack == 4)
    {
        auto weight_data_packed_ra = weight_data_packed.raw_accessor<float, 3>();
        {
            otter::parallel_for(0, outh, 0, [&](int64_t begin, int64_t end) {
                for (const auto p : otter::irange(begin, end))
                {
                    float* outptr = top_blob_ra[p].data();

                    for (int j = 0; j < outw; j++)
                    {
                        __m128 _sum = _mm_set1_ps(0.f);

                        if (bias_term)
                        {
                            _sum = _mm_loadu_ps((const float*)bias_data_ptr + p * 4);
                        }

                        const float* kptr = weight_data_packed_ra[p].data();

                        for (int q = 0; q < h; q++)
                        {
                            const float* sptr = bottom_blob_bordered_ra[q].data() + j * stride_w * 4;

                            for (int k = 0; k < kernel_w; k++)
                            {
                                __m128 _val0 = _mm_set1_ps(sptr[0]);
                                __m128 _val1 = _mm_set1_ps(sptr[1]);
                                __m128 _val2 = _mm_set1_ps(sptr[2]);
                                __m128 _val3 = _mm_set1_ps(sptr[3]);

                                __m128 _w0 = _mm_loadu_ps(kptr);
                                _sum = _mm_add_ps(_mm_mul_ps(_val0, _w0), _sum);
                                __m128 _w1 = _mm_loadu_ps(kptr + 4);
                                _sum = _mm_add_ps(_mm_mul_ps(_val1, _w1), _sum);
                                __m128 _w2 = _mm_loadu_ps(kptr + 8);
                                _sum = _mm_add_ps(_mm_mul_ps(_val2, _w2), _sum);
                                __m128 _w3 = _mm_loadu_ps(kptr + 12);
                                _sum = _mm_add_ps(_mm_mul_ps(_val3, _w3), _sum);

                                sptr += dilation_w * 4;
                                kptr += 16;
                            }
                        }

                        _sum = activation_sse(_sum, activation_type, activation_params);

                        _mm_storeu_ps(outptr, _sum);
                        outptr += 4;
                    }
                }
            });
        }
    }

    if (elempack == 1 && out_elempack == 4)
    {
        auto weight_data_packed_ra = weight_data_packed.raw_accessor<float, 3>();
        {
            otter::parallel_for(0, outh, 0, [&](int64_t begin, int64_t end) {
                for (const auto p : otter::irange(begin, end))
                {
                    float* outptr = top_blob_ra[p].data();

                    for (int j = 0; j < outw; j++)
                    {
                        __m128 _sum = _mm_set1_ps(0.f);

                        if (bias_term)
                        {
                            _sum = _mm_loadu_ps((const float*)bias_data_ptr + p * 4);
                        }

                        const float* kptr = weight_data_packed_ra[p].data();

                        for (int q = 0; q < h; q++)
                        {
                            const float* sptr = bottom_blob_bordered_ra[q].data() + j * stride_w;

                            for (int k = 0; k < kernel_w; k++)
                            {
                                __m128 _val = _mm_set1_ps(sptr[0]);
                                __m128 _w = _mm_loadu_ps(kptr);
                                _sum = _mm_add_ps(_mm_mul_ps(_val, _w), _sum);

                                sptr += dilation_w;
                                kptr += 4;
                            }
                        }

                        _sum = activation_sse(_sum, activation_type, activation_params);

                        _mm_storeu_ps(outptr, _sum);
                        outptr += 4;
                    }
                }
            });
        }
    }

    if (elempack == 4 && out_elempack == 1)
    {
        auto weight_data_packed_ra = weight_data_packed.raw_accessor<float, 3>();
        {
            otter::parallel_for(0, outh, 0, [&](int64_t begin, int64_t end) {
                for (const auto p : otter::irange(begin, end))
                {
                    float* outptr = top_blob_ra[p].data();

                    for (int j = 0; j < outw; j++)
                    {
                        float sum = 0.f;

                        if (bias_term)
                        {
                            sum = bias_data_ptr[p];
                        }

                        const float* kptr = weight_data_packed_ra[p].data();

                        for (int q = 0; q < h; q++)
                        {
                            const float* sptr = bottom_blob_bordered_ra[q].data() + j * stride_w * 4;

                            for (int k = 0; k < kernel_w; k++)
                            {
                                __m128 _val = _mm_loadu_ps(sptr);
                                __m128 _w = _mm_loadu_ps(kptr);
                                __m128 _s4 = _mm_mul_ps(_val, _w);
                                sum += _mm_reduce_add_ps(_s4); // dot

                                sptr += dilation_w * 4;
                                kptr += 4;
                            }
                        }

                        sum = activation_ss(sum, activation_type, activation_params);

                        outptr[j] = sum;
                    }
                }
            });
        }
    }
#endif // __SSE2__
    
    const float* weight_data_ptr = weight_data.data_ptr<float>();

    if (elempack == 1 && out_elempack == 1)
    {
        {
            otter::parallel_for(0, outh, 0, [&](int64_t begin, int64_t end) {
                for (const auto p : otter::irange(begin, end)) {
                    float* outptr = top_blob_ra[p].data();

                    for (int j = 0; j < outw; j++)
                    {
                        float sum = 0.f;

                        if (bias_term)
                        {
                            sum = bias_data_ptr[p];
                        }

                        const float* kptr = (const float*)weight_data_ptr + kernel_w * h * p;

                        for (int q = 0; q < h; q++)
                        {
                            const float* sptr = bottom_blob_bordered_ra[q].data() + j * stride_w;

                            for (int k = 0; k < kernel_w; k++)
                            {
                                float val = sptr[0];
                                float wt = kptr[0];
                                sum += val * wt;

                                sptr += dilation_w;
                                kptr += 1;
                            }
                        }

                        sum = activation_ss(sum, activation_type, activation_params);

                        outptr[j] = sum;
                    }
                }
            });
        }
    }

    return 0;
}

}   // end namespace otter
