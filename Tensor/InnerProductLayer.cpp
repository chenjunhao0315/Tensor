//
//  InnerProductLayer.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/7/7.
//

#include "InnerProductLayer.hpp"
#include "Tensor.hpp"
#include "TensorMaker.hpp"
#include "TensorFactory.hpp"
#include "Parallel.hpp"

#include "QuantizeX86.hpp"
#include "ActivationLayer.hpp"
#include "TensorPacking.hpp"

namespace otter {

InnerProductLayer::InnerProductLayer() {
    one_blob_only = true;
    support_inplace = false;
#if __SSE2__
    support_packing = true;
#endif
}

InnerProductLayer::~InnerProductLayer() {
    if (activation) {
        delete activation;
        activation = nullptr;
    }
}

int InnerProductLayer::parse_param(LayerOption& option, ParamDict& pd) {
    int out_features = opt_find_int(option, "out_features", 0);
    int bias_term = 0;
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
    
    pd.set((int)InnerProductParam::OutFeatures, out_features);
    pd.set((int)InnerProductParam::Bias_term, bias_term);
    pd.set((int)InnerProductParam::Activation_type, activation_type);
    pd.set((int)InnerProductParam::Activation_params, activation_params);
    
    return 0;
}

int InnerProductLayer::compute_output_shape(ParamDict& pd) {
    auto shape_a = bottom_shapes[0].accessor<int, 2>()[0];
    int dims = bottom_shapes[0][0].numel();
    
    int out_features = pd.get((int)InnerProductParam::OutFeatures, 0);
    
    if (dims == 1) {
        int in_feautres = shape_a[0];
        pd.set(OUTPUT_SHAPE_HINT, otter::tensor({out_features}, ScalarType::Int).view({1, -1}));
        pd.set((int)InnerProductParam::InFeatures, in_feautres);
    } else if (dims == 2) {
        int h = shape_a[0];
        int w = shape_a[1];
        pd.set(OUTPUT_SHAPE_HINT, otter::tensor({h, out_features}, ScalarType::Int).view({1, -1}));
        pd.set((int)InnerProductParam::InFeatures, w);
    } else {
        OTTER_CHECK(false, "InnerProduct shape error!");
    }
    
    return 0;
}

int InnerProductLayer::load_param(const ParamDict& pd) {
    out_features = pd.get((int)InnerProductParam::OutFeatures, 0);
    in_features  = pd.get((int)InnerProductParam::InFeatures, 0);
    bias_term    = pd.get((int)InnerProductParam::Bias_term, 0);
    activation_type = pd.get((int)InnerProductParam::Activation_type, 0);
    activation_params = pd.get((int)InnerProductParam::Activation_params, Tensor());
    
    return 0;
}

int InnerProductLayer::init_model() {
    weight_data = otter::rand({out_features, in_features}, otter::ScalarType::Float);
    
    if (bias_term)
        bias_data = otter::rand({out_features}, ScalarType::Float);
    
    return 0;
}

int InnerProductLayer::load_model(const Initializer& initializer) {
    weight_data = initializer.load({out_features, in_features}, 0);
    
    if (bias_term) {
        bias_data = initializer.load({out_features}, 1);
    }
    
    return 0;
}

int InnerProductLayer::create_pipeline(const NetOption& opt) {
    activation = create_activation_layer(activation_type, activation_params);

    int out_elempack = 1;

#if __SSE2__
    if (opt.use_packing_layout) {
#if __AVX512F__
        out_elempack = out_features % 16 == 0 ? 16 : out_features % 8 == 0 ? 8 : out_features % 4 == 0 ? 4 : 1;
#elif __AVX__
        out_elempack = out_features % 8 == 0 ? 8 : out_features % 4 == 0 ? 4 : 1;
#else
        out_elempack = out_features % 4 == 0 ? 4 : 1;
#endif
    }
#endif // __SSE2__

    if (out_elempack != 1) {
        // src = inch-outch
        // dst = pb-inch-outch/pb
        {
            Tensor weight_data_r2 = weight_data.view({out_features, in_features});
            
            weight_data_tm = otter::empty({out_features / out_elempack, in_features}, otter::get_update_scalarType(otter::ScalarType::Float, out_elempack));
            
            auto weight_data_tm_ra = weight_data_tm.raw_accessor<float, 2>();
            auto weight_data_r2_a = weight_data.accessor<float, 2>();

            for (int q = 0; q + (out_elempack - 1) < out_features; q += out_elempack) {
                float* g0 = weight_data_tm_ra[q / out_elempack].data();

                for (int p = 0; p < in_features; p++) {
                    for (int j = 0; j < out_elempack; j++) {
                        *g0++ = weight_data_r2_a[q + j][p];
                    }
                }
            }
        }
    } else {
        weight_data_tm = weight_data;
    }
    
    return 0;
}

int InnerProductLayer::forward(const Tensor &bottom_blob, Tensor &top_blob, const NetOption &opt) const {
    
    if (bottom_blob.dim() == 2 && bottom_blob.size(1) == in_features && bottom_blob.size(0) * bottom_blob.elempack() > 1) {
        // gemm
        int h = bottom_blob.size(0);
        auto dtype = bottom_blob.scalar_type();
        int elempack = bottom_blob.elempack();

        top_blob = otter::empty({h, out_features}, otter::get_update_scalarType(dtype, elempack));

        int out_features_elempack = 1;
#if __SSE2__
        if (opt.use_packing_layout)
        {
#if __AVX__
            out_features_elempack = out_features % 8 == 0 ? 8 : out_features % 4 == 0 ? 4 : 1;
#else
            out_features_elempack = out_features % 4 == 0 ? 4 : 1;
#endif
        }
#endif // __SSE2__
        
        auto bottom_blob_ra = bottom_blob.raw_accessor<float, 2>();
        auto top_blob_ra = top_blob.raw_accessor<float, 2>();
        auto weight_data_tm_ra = weight_data_tm.raw_accessor<float, 2>();
        const float* bias_data_ptr = (const float*)bias_data.data_ptr();

        otter::parallel_for(0, h, 0, [&](int64_t begin, int64_t end) {
            for (const auto j : otter::irange(begin, end)) {
    #if __SSE2__
    #if __AVX__
                if (elempack == 8 && out_features_elempack == 8) {
                    float* outptr = top_blob_ra[j].data();

                    for (int p = 0; p < out_features / out_features_elempack; p++) {
                        const float* kptr = weight_data_tm_ra[p].data();
                        const float* m = bottom_blob_ra[j].data();

                        __m256 _sum0 = _mm256_set1_ps(0.f);
                        __m256 _sum1 = _mm256_set1_ps(0.f);
                        __m256 _sum2 = _mm256_set1_ps(0.f);
                        __m256 _sum3 = _mm256_set1_ps(0.f);
                        __m256 _sum4 = _mm256_set1_ps(0.f);
                        __m256 _sum5 = _mm256_set1_ps(0.f);
                        __m256 _sum6 = _mm256_set1_ps(0.f);
                        __m256 _sum7 = _mm256_set1_ps(0.f);

                        if (bias_term) {
                            _sum0 = _mm256_set1_ps(bias_data_ptr[p * 8 + 0]);
                            _sum1 = _mm256_set1_ps(bias_data_ptr[p * 8 + 1]);
                            _sum2 = _mm256_set1_ps(bias_data_ptr[p * 8 + 2]);
                            _sum3 = _mm256_set1_ps(bias_data_ptr[p * 8 + 3]);
                            _sum4 = _mm256_set1_ps(bias_data_ptr[p * 8 + 4]);
                            _sum5 = _mm256_set1_ps(bias_data_ptr[p * 8 + 5]);
                            _sum6 = _mm256_set1_ps(bias_data_ptr[p * 8 + 6]);
                            _sum7 = _mm256_set1_ps(bias_data_ptr[p * 8 + 7]);
                        }

                        for (int i = 0; i < in_features; i++) {
                            __m256 _val = _mm256_loadu_ps(m);
                            __m256 _k0 = _mm256_set1_ps(kptr[0]);
                            __m256 _k1 = _mm256_set1_ps(kptr[1]);
                            __m256 _k2 = _mm256_set1_ps(kptr[2]);
                            __m256 _k3 = _mm256_set1_ps(kptr[3]);
                            __m256 _k4 = _mm256_set1_ps(kptr[4]);
                            __m256 _k5 = _mm256_set1_ps(kptr[5]);
                            __m256 _k6 = _mm256_set1_ps(kptr[6]);
                            __m256 _k7 = _mm256_set1_ps(kptr[7]);
                            _sum0 = _mm256_comp_fmadd_ps(_val, _k0, _sum0);
                            _sum1 = _mm256_comp_fmadd_ps(_val, _k1, _sum1);
                            _sum2 = _mm256_comp_fmadd_ps(_val, _k2, _sum2);
                            _sum3 = _mm256_comp_fmadd_ps(_val, _k3, _sum3);
                            _sum4 = _mm256_comp_fmadd_ps(_val, _k4, _sum4);
                            _sum5 = _mm256_comp_fmadd_ps(_val, _k5, _sum5);
                            _sum6 = _mm256_comp_fmadd_ps(_val, _k6, _sum6);
                            _sum7 = _mm256_comp_fmadd_ps(_val, _k7, _sum7);

                            m += 8;
                            kptr += 8;
                        }

                        _sum0 = activation_avx(_sum0, activation_type, activation_params);
                        _sum1 = activation_avx(_sum1, activation_type, activation_params);
                        _sum2 = activation_avx(_sum2, activation_type, activation_params);
                        _sum3 = activation_avx(_sum3, activation_type, activation_params);
                        _sum4 = activation_avx(_sum4, activation_type, activation_params);
                        _sum5 = activation_avx(_sum5, activation_type, activation_params);
                        _sum6 = activation_avx(_sum6, activation_type, activation_params);
                        _sum7 = activation_avx(_sum7, activation_type, activation_params);

                        _mm256_storeu_ps(outptr, _sum0);
                        _mm256_storeu_ps(outptr + 8, _sum1);
                        _mm256_storeu_ps(outptr + 16, _sum2);
                        _mm256_storeu_ps(outptr + 24, _sum3);
                        _mm256_storeu_ps(outptr + 32, _sum4);
                        _mm256_storeu_ps(outptr + 40, _sum5);
                        _mm256_storeu_ps(outptr + 48, _sum6);
                        _mm256_storeu_ps(outptr + 56, _sum7);
                        outptr += 64;
                    }
                }

                if (elempack == 1 && out_features_elempack == 8) {
                    float* outptr = top_blob_ra[j].data();

                    for (int p = 0; p < out_features / out_features_elempack; p++) {
                        const float* kptr = weight_data_tm_ra[p].data();
                        const float* m = bottom_blob_ra[j].data();

                        __m256 _sum = _mm256_set1_ps(0.f);

                        if (bias_term) {
                            _sum = _mm256_loadu_ps((const float*)bias_data_ptr + p * 8);
                        }

                        int i = 0;
                        for (; i + 7 < in_features; i += 8) {
                            __m256 _val0 = _mm256_broadcast_ss(m);
                            __m256 _val1 = _mm256_broadcast_ss(m + 1);
                            __m256 _val2 = _mm256_broadcast_ss(m + 2);
                            __m256 _val3 = _mm256_broadcast_ss(m + 3);
                            __m256 _val4 = _mm256_broadcast_ss(m + 4);
                            __m256 _val5 = _mm256_broadcast_ss(m + 5);
                            __m256 _val6 = _mm256_broadcast_ss(m + 6);
                            __m256 _val7 = _mm256_broadcast_ss(m + 7);

                            __m256 _w0 = _mm256_loadu_ps(kptr);
                            _sum = _mm256_comp_fmadd_ps(_val0, _w0, _sum);
                            __m256 _w1 = _mm256_loadu_ps(kptr + 8);
                            _sum = _mm256_comp_fmadd_ps(_val1, _w1, _sum);
                            __m256 _w2 = _mm256_loadu_ps(kptr + 16);
                            _sum = _mm256_comp_fmadd_ps(_val2, _w2, _sum);
                            __m256 _w3 = _mm256_loadu_ps(kptr + 24);
                            _sum = _mm256_comp_fmadd_ps(_val3, _w3, _sum);
                            __m256 _w4 = _mm256_loadu_ps(kptr + 32);
                            _sum = _mm256_comp_fmadd_ps(_val4, _w4, _sum);
                            __m256 _w5 = _mm256_loadu_ps(kptr + 40);
                            _sum = _mm256_comp_fmadd_ps(_val5, _w5, _sum);
                            __m256 _w6 = _mm256_loadu_ps(kptr + 48);
                            _sum = _mm256_comp_fmadd_ps(_val6, _w6, _sum);
                            __m256 _w7 = _mm256_loadu_ps(kptr + 56);
                            _sum = _mm256_comp_fmadd_ps(_val7, _w7, _sum);

                            m += 8;
                            kptr += 64;
                        }
                        for (; i + 3 < in_features; i += 4) {
                            __m256 _val0 = _mm256_broadcast_ss(m);
                            __m256 _val1 = _mm256_broadcast_ss(m + 1);
                            __m256 _val2 = _mm256_broadcast_ss(m + 2);
                            __m256 _val3 = _mm256_broadcast_ss(m + 3);

                            __m256 _w0 = _mm256_loadu_ps(kptr);
                            _sum = _mm256_comp_fmadd_ps(_val0, _w0, _sum);
                            __m256 _w1 = _mm256_loadu_ps(kptr + 8);
                            _sum = _mm256_comp_fmadd_ps(_val1, _w1, _sum);
                            __m256 _w2 = _mm256_loadu_ps(kptr + 16);
                            _sum = _mm256_comp_fmadd_ps(_val2, _w2, _sum);
                            __m256 _w3 = _mm256_loadu_ps(kptr + 24);
                            _sum = _mm256_comp_fmadd_ps(_val3, _w3, _sum);

                            m += 4;
                            kptr += 32;
                        }
                        for (; i < in_features; i++) {
                            __m256 _val = _mm256_set1_ps(m[0]);
                            __m256 _w = _mm256_loadu_ps(kptr);
                            _sum = _mm256_comp_fmadd_ps(_val, _w, _sum);

                            m += 1;
                            kptr += 8;
                        }

                        _sum = activation_avx(_sum, activation_type, activation_params);

                        _mm256_storeu_ps(outptr, _sum);
                        outptr += 8;
                    }
                }

                if (elempack == 4 && out_features_elempack == 8) {
                    float* outptr = top_blob_ra[j].data();

                    for (int p = 0; p < out_features / out_features_elempack; p++) {
                        const float* kptr = weight_data_tm_ra[p].data();
                        const float* m = bottom_blob_ra[j].data();

                        __m128 _sum0 = _mm_set1_ps(0.f);
                        __m128 _sum1 = _mm_set1_ps(0.f);
                        __m128 _sum2 = _mm_set1_ps(0.f);
                        __m128 _sum3 = _mm_set1_ps(0.f);
                        __m128 _sum4 = _mm_set1_ps(0.f);
                        __m128 _sum5 = _mm_set1_ps(0.f);
                        __m128 _sum6 = _mm_set1_ps(0.f);
                        __m128 _sum7 = _mm_set1_ps(0.f);

                        if (bias_term) {
                            _sum0 = _mm_set1_ps(bias_data_ptr[p * 8 + 0]);
                            _sum1 = _mm_set1_ps(bias_data_ptr[p * 8 + 1]);
                            _sum2 = _mm_set1_ps(bias_data_ptr[p * 8 + 2]);
                            _sum3 = _mm_set1_ps(bias_data_ptr[p * 8 + 3]);
                            _sum4 = _mm_set1_ps(bias_data_ptr[p * 8 + 4]);
                            _sum5 = _mm_set1_ps(bias_data_ptr[p * 8 + 5]);
                            _sum6 = _mm_set1_ps(bias_data_ptr[p * 8 + 6]);
                            _sum7 = _mm_set1_ps(bias_data_ptr[p * 8 + 7]);
                        }

                        int i = 0;
                        for (; i < in_features; i++) {
                            __m128 _val = _mm_loadu_ps(m);
                            _sum0 = _mm_comp_fmadd_ps(_val, _mm_set1_ps(kptr[0]), _sum0);
                            _sum1 = _mm_comp_fmadd_ps(_val, _mm_set1_ps(kptr[1]), _sum1);
                            _sum2 = _mm_comp_fmadd_ps(_val, _mm_set1_ps(kptr[2]), _sum2);
                            _sum3 = _mm_comp_fmadd_ps(_val, _mm_set1_ps(kptr[3]), _sum3);
                            _sum4 = _mm_comp_fmadd_ps(_val, _mm_set1_ps(kptr[4]), _sum4);
                            _sum5 = _mm_comp_fmadd_ps(_val, _mm_set1_ps(kptr[5]), _sum5);
                            _sum6 = _mm_comp_fmadd_ps(_val, _mm_set1_ps(kptr[6]), _sum6);
                            _sum7 = _mm_comp_fmadd_ps(_val, _mm_set1_ps(kptr[7]), _sum7);

                            m += 4;
                            kptr += 8;
                        }

                        _sum0 = activation_sse(_sum0, activation_type, activation_params);
                        _sum1 = activation_sse(_sum1, activation_type, activation_params);
                        _sum2 = activation_sse(_sum2, activation_type, activation_params);
                        _sum3 = activation_sse(_sum3, activation_type, activation_params);
                        _sum4 = activation_sse(_sum4, activation_type, activation_params);
                        _sum5 = activation_sse(_sum5, activation_type, activation_params);
                        _sum6 = activation_sse(_sum6, activation_type, activation_params);
                        _sum7 = activation_sse(_sum7, activation_type, activation_params);

                        _mm_storeu_ps(outptr, _sum0);
                        _mm_storeu_ps(outptr + 4, _sum1);
                        _mm_storeu_ps(outptr + 8, _sum2);
                        _mm_storeu_ps(outptr + 12, _sum3);
                        _mm_storeu_ps(outptr + 16, _sum4);
                        _mm_storeu_ps(outptr + 20, _sum5);
                        _mm_storeu_ps(outptr + 24, _sum6);
                        _mm_storeu_ps(outptr + 28, _sum7);
                        outptr += 32;
                    }
                }

                if (elempack == 8 && out_features_elempack == 1) {
                    float* outptr = top_blob_ra[j].data();

                    for (int p = 0; p < out_features; p++) {
                        const float* kptr = (const float*)weight_data_tm_ra.data() + in_features * p;
                        const float* m = bottom_blob_ra[j].data();

                        __m256 _sum0 = _mm256_set1_ps(0.f);
                        __m256 _sum1 = _mm256_set1_ps(0.f);
                        __m256 _sum2 = _mm256_set1_ps(0.f);
                        __m256 _sum3 = _mm256_set1_ps(0.f);

                        if (bias_term) {
                            _sum0 = _mm256_set1_ps(bias_data_ptr[p]);
                        }

                        int i = 0;
                        for (; i + 7 < in_features; i += 8) {
                            __m256 _val0 = _mm256_loadu_ps(m);
                            __m256 _val1 = _mm256_loadu_ps(m + 8);
                            __m256 _val2 = _mm256_loadu_ps(m + 16);
                            __m256 _val3 = _mm256_loadu_ps(m + 24);
                            __m256 _val4 = _mm256_loadu_ps(m + 32);
                            __m256 _val5 = _mm256_loadu_ps(m + 40);
                            __m256 _val6 = _mm256_loadu_ps(m + 48);
                            __m256 _val7 = _mm256_loadu_ps(m + 56);
                            _sum0 = _mm256_comp_fmadd_ps(_val0, _mm256_set1_ps(kptr[0]), _sum0);
                            _sum1 = _mm256_comp_fmadd_ps(_val1, _mm256_set1_ps(kptr[1]), _sum1);
                            _sum2 = _mm256_comp_fmadd_ps(_val2, _mm256_set1_ps(kptr[2]), _sum2);
                            _sum3 = _mm256_comp_fmadd_ps(_val3, _mm256_set1_ps(kptr[3]), _sum3);
                            _sum0 = _mm256_comp_fmadd_ps(_val4, _mm256_set1_ps(kptr[4]), _sum0);
                            _sum1 = _mm256_comp_fmadd_ps(_val5, _mm256_set1_ps(kptr[5]), _sum1);
                            _sum2 = _mm256_comp_fmadd_ps(_val6, _mm256_set1_ps(kptr[6]), _sum2);
                            _sum3 = _mm256_comp_fmadd_ps(_val7, _mm256_set1_ps(kptr[7]), _sum3);

                            m += 64;
                            kptr += 8;
                        }
                        for (; i + 3 < in_features; i += 4) {
                            __m256 _val0 = _mm256_loadu_ps(m);
                            __m256 _val1 = _mm256_loadu_ps(m + 8);
                            __m256 _val2 = _mm256_loadu_ps(m + 16);
                            __m256 _val3 = _mm256_loadu_ps(m + 24);
                            _sum0 = _mm256_comp_fmadd_ps(_val0, _mm256_set1_ps(kptr[0]), _sum0);
                            _sum1 = _mm256_comp_fmadd_ps(_val1, _mm256_set1_ps(kptr[1]), _sum1);
                            _sum2 = _mm256_comp_fmadd_ps(_val2, _mm256_set1_ps(kptr[2]), _sum2);
                            _sum3 = _mm256_comp_fmadd_ps(_val3, _mm256_set1_ps(kptr[3]), _sum3);

                            m += 32;
                            kptr += 4;
                        }
                        for (; i < in_features; i++) {
                            __m256 _val = _mm256_loadu_ps(m);
                            __m256 _k = _mm256_set1_ps(kptr[0]);
                            _sum0 = _mm256_comp_fmadd_ps(_val, _k, _sum0);

                            m += 8;
                            kptr += 1;
                        }

                        _sum0 = _mm256_add_ps(_sum0, _sum1);
                        _sum2 = _mm256_add_ps(_sum2, _sum3);
                        _sum0 = _mm256_add_ps(_sum0, _sum2);

                        _sum0 = activation_avx(_sum0, activation_type, activation_params);

                        _mm256_storeu_ps(outptr, _sum0);
                        outptr += 8;
                    }
                }

                if (elempack == 8 && out_features_elempack == 4) {
                    float* outptr = top_blob_ra[j].data();

                    for (int p = 0; p < out_features / out_features_elempack; p++) {
                        const float* kptr = weight_data_tm_ra[p].data();
                        const float* m = bottom_blob_ra[j].data();

                        __m256 _sum0 = _mm256_set1_ps(0.f);
                        __m256 _sum1 = _mm256_set1_ps(0.f);
                        __m256 _sum2 = _mm256_set1_ps(0.f);
                        __m256 _sum3 = _mm256_set1_ps(0.f);

                        if (bias_term) {
                            _sum0 = _mm256_set1_ps(bias_data_ptr[p * 4 + 0]);
                            _sum1 = _mm256_set1_ps(bias_data_ptr[p * 4 + 1]);
                            _sum2 = _mm256_set1_ps(bias_data_ptr[p * 4 + 2]);
                            _sum3 = _mm256_set1_ps(bias_data_ptr[p * 4 + 3]);
                        }

                        int i = 0;
                        for (; i + 3 < in_features; i += 4) {
                            __m256 _val0 = _mm256_loadu_ps(m);
                            __m256 _val1 = _mm256_loadu_ps(m + 8);
                            __m256 _val2 = _mm256_loadu_ps(m + 16);
                            __m256 _val3 = _mm256_loadu_ps(m + 24);
                            _sum0 = _mm256_comp_fmadd_ps(_val0, _mm256_set1_ps(kptr[0]), _sum0);
                            _sum1 = _mm256_comp_fmadd_ps(_val0, _mm256_set1_ps(kptr[1]), _sum1);
                            _sum2 = _mm256_comp_fmadd_ps(_val0, _mm256_set1_ps(kptr[2]), _sum2);
                            _sum3 = _mm256_comp_fmadd_ps(_val0, _mm256_set1_ps(kptr[3]), _sum3);
                            _sum0 = _mm256_comp_fmadd_ps(_val1, _mm256_set1_ps(kptr[4]), _sum0);
                            _sum1 = _mm256_comp_fmadd_ps(_val1, _mm256_set1_ps(kptr[5]), _sum1);
                            _sum2 = _mm256_comp_fmadd_ps(_val1, _mm256_set1_ps(kptr[6]), _sum2);
                            _sum3 = _mm256_comp_fmadd_ps(_val1, _mm256_set1_ps(kptr[7]), _sum3);
                            kptr += 8;

                            _sum0 = _mm256_comp_fmadd_ps(_val2, _mm256_set1_ps(kptr[0]), _sum0);
                            _sum1 = _mm256_comp_fmadd_ps(_val2, _mm256_set1_ps(kptr[1]), _sum1);
                            _sum2 = _mm256_comp_fmadd_ps(_val2, _mm256_set1_ps(kptr[2]), _sum2);
                            _sum3 = _mm256_comp_fmadd_ps(_val2, _mm256_set1_ps(kptr[3]), _sum3);
                            _sum0 = _mm256_comp_fmadd_ps(_val3, _mm256_set1_ps(kptr[4]), _sum0);
                            _sum1 = _mm256_comp_fmadd_ps(_val3, _mm256_set1_ps(kptr[5]), _sum1);
                            _sum2 = _mm256_comp_fmadd_ps(_val3, _mm256_set1_ps(kptr[6]), _sum2);
                            _sum3 = _mm256_comp_fmadd_ps(_val3, _mm256_set1_ps(kptr[7]), _sum3);

                            m += 32;
                            kptr += 8;
                        }
                        for (; i < in_features; i++) {
                            __m256 _val = _mm256_loadu_ps(m);
                            _sum0 = _mm256_comp_fmadd_ps(_val, _mm256_set1_ps(kptr[0]), _sum0);
                            _sum1 = _mm256_comp_fmadd_ps(_val, _mm256_set1_ps(kptr[1]), _sum1);
                            _sum2 = _mm256_comp_fmadd_ps(_val, _mm256_set1_ps(kptr[2]), _sum2);
                            _sum3 = _mm256_comp_fmadd_ps(_val, _mm256_set1_ps(kptr[3]), _sum3);

                            m += 8;
                            kptr += 4;
                        }

                        _sum0 = activation_avx(_sum0, activation_type, activation_params);
                        _sum1 = activation_avx(_sum1, activation_type, activation_params);
                        _sum2 = activation_avx(_sum2, activation_type, activation_params);
                        _sum3 = activation_avx(_sum3, activation_type, activation_params);

                        _mm256_storeu_ps(outptr, _sum0);
                        _mm256_storeu_ps(outptr + 8, _sum1);
                        _mm256_storeu_ps(outptr + 16, _sum2);
                        _mm256_storeu_ps(outptr + 24, _sum3);
                        outptr += 32;
                    }
                }
    #endif // __AVX__

                if (elempack == 4 && out_features_elempack == 4) {
                    float* outptr = top_blob_ra[j].data();

                    for (int p = 0; p < out_features / out_features_elempack; p++) {
                        const float* kptr = weight_data_tm_ra[p].data();
                        const float* m = bottom_blob_ra[j].data();

                        __m128 _sum0 = _mm_set1_ps(0.f);
                        __m128 _sum1 = _mm_set1_ps(0.f);
                        __m128 _sum2 = _mm_set1_ps(0.f);
                        __m128 _sum3 = _mm_set1_ps(0.f);

                        if (bias_term) {
                            _sum0 = _mm_set1_ps(bias_data_ptr[p * 4 + 0]);
                            _sum1 = _mm_set1_ps(bias_data_ptr[p * 4 + 1]);
                            _sum2 = _mm_set1_ps(bias_data_ptr[p * 4 + 2]);
                            _sum3 = _mm_set1_ps(bias_data_ptr[p * 4 + 3]);
                        }

                        int i = 0;
                        for (; i + 3 < in_features; i += 4) {
                            __m128 _val0 = _mm_loadu_ps(m);
                            __m128 _val1 = _mm_loadu_ps(m + 4);
                            __m128 _val2 = _mm_loadu_ps(m + 8);
                            __m128 _val3 = _mm_loadu_ps(m + 12);
                            _sum0 = _mm_add_ps(_mm_mul_ps(_val0, _mm_set1_ps(kptr[0])), _sum0);
                            _sum1 = _mm_add_ps(_mm_mul_ps(_val0, _mm_set1_ps(kptr[1])), _sum1);
                            _sum2 = _mm_add_ps(_mm_mul_ps(_val0, _mm_set1_ps(kptr[2])), _sum2);
                            _sum3 = _mm_add_ps(_mm_mul_ps(_val0, _mm_set1_ps(kptr[3])), _sum3);
                            _sum0 = _mm_add_ps(_mm_mul_ps(_val1, _mm_set1_ps(kptr[4])), _sum0);
                            _sum1 = _mm_add_ps(_mm_mul_ps(_val1, _mm_set1_ps(kptr[5])), _sum1);
                            _sum2 = _mm_add_ps(_mm_mul_ps(_val1, _mm_set1_ps(kptr[6])), _sum2);
                            _sum3 = _mm_add_ps(_mm_mul_ps(_val1, _mm_set1_ps(kptr[7])), _sum3);
                            _sum0 = _mm_add_ps(_mm_mul_ps(_val2, _mm_set1_ps(kptr[8])), _sum0);
                            _sum1 = _mm_add_ps(_mm_mul_ps(_val2, _mm_set1_ps(kptr[9])), _sum1);
                            _sum2 = _mm_add_ps(_mm_mul_ps(_val2, _mm_set1_ps(kptr[10])), _sum2);
                            _sum3 = _mm_add_ps(_mm_mul_ps(_val2, _mm_set1_ps(kptr[11])), _sum3);
                            _sum0 = _mm_add_ps(_mm_mul_ps(_val3, _mm_set1_ps(kptr[12])), _sum0);
                            _sum1 = _mm_add_ps(_mm_mul_ps(_val3, _mm_set1_ps(kptr[13])), _sum1);
                            _sum2 = _mm_add_ps(_mm_mul_ps(_val3, _mm_set1_ps(kptr[14])), _sum2);
                            _sum3 = _mm_add_ps(_mm_mul_ps(_val3, _mm_set1_ps(kptr[15])), _sum3);

                            m += 16;
                            kptr += 16;
                        }
                        for (; i < in_features; i++) {
                            __m128 _val = _mm_loadu_ps(m);
                            _sum0 = _mm_add_ps(_mm_mul_ps(_val, _mm_set1_ps(kptr[0])), _sum0);
                            _sum1 = _mm_add_ps(_mm_mul_ps(_val, _mm_set1_ps(kptr[1])), _sum1);
                            _sum2 = _mm_add_ps(_mm_mul_ps(_val, _mm_set1_ps(kptr[2])), _sum2);
                            _sum3 = _mm_add_ps(_mm_mul_ps(_val, _mm_set1_ps(kptr[3])), _sum3);

                            m += 4;
                            kptr += 4;
                        }

                        _sum0 = activation_sse(_sum0, activation_type, activation_params);
                        _sum1 = activation_sse(_sum1, activation_type, activation_params);
                        _sum2 = activation_sse(_sum2, activation_type, activation_params);
                        _sum3 = activation_sse(_sum3, activation_type, activation_params);

                        _mm_storeu_ps(outptr, _sum0);
                        _mm_storeu_ps(outptr + 4, _sum1);
                        _mm_storeu_ps(outptr + 8, _sum2);
                        _mm_storeu_ps(outptr + 12, _sum3);
                        outptr += 16;
                    }
                }

                if (elempack == 1 && out_features_elempack == 4) {
                    float* outptr = top_blob_ra[j].data();

                    for (int p = 0; p < out_features / out_features_elempack; p++) {
                        const float* kptr = weight_data_tm_ra[p].data();
                        const float* m = bottom_blob_ra[j].data();

                        __m128 _sum = _mm_set1_ps(0.f);

                        if (bias_term) {
                            _sum = _mm_loadu_ps((const float*)bias_data_ptr + p * 4);
                        }

                        int i = 0;
    #if __AVX__
                        for (; i + 7 < in_features; i += 8) {
                            __m128 _val0 = _mm_broadcast_ss(m);
                            __m128 _val1 = _mm_broadcast_ss(m + 1);
                            __m128 _val2 = _mm_broadcast_ss(m + 2);
                            __m128 _val3 = _mm_broadcast_ss(m + 3);
                            __m128 _val4 = _mm_broadcast_ss(m + 4);
                            __m128 _val5 = _mm_broadcast_ss(m + 5);
                            __m128 _val6 = _mm_broadcast_ss(m + 6);
                            __m128 _val7 = _mm_broadcast_ss(m + 7);

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

                            m += 8;
                            kptr += 32;
                        }
    #endif // __AVX__
                        for (; i + 3 < in_features; i += 4) {
                            __m128 _val0 = _mm_set1_ps(m[0]);
                            __m128 _val1 = _mm_set1_ps(m[1]);
                            __m128 _val2 = _mm_set1_ps(m[2]);
                            __m128 _val3 = _mm_set1_ps(m[3]);

                            __m128 _w0 = _mm_loadu_ps(kptr);
                            _sum = _mm_add_ps(_mm_mul_ps(_val0, _w0), _sum);
                            __m128 _w1 = _mm_loadu_ps(kptr + 4);
                            _sum = _mm_add_ps(_mm_mul_ps(_val1, _w1), _sum);
                            __m128 _w2 = _mm_loadu_ps(kptr + 8);
                            _sum = _mm_add_ps(_mm_mul_ps(_val2, _w2), _sum);
                            __m128 _w3 = _mm_loadu_ps(kptr + 12);
                            _sum = _mm_add_ps(_mm_mul_ps(_val3, _w3), _sum);

                            m += 4;
                            kptr += 16;
                        }
                        for (; i < in_features; i++) {
                            __m128 _val = _mm_set1_ps(m[0]);
                            __m128 _k = _mm_loadu_ps(kptr);
                            _sum = _mm_add_ps(_mm_mul_ps(_val, _k), _sum);

                            m += 1;
                            kptr += 4;
                        }

                        _sum = activation_sse(_sum, activation_type, activation_params);

                        _mm_storeu_ps(outptr, _sum);
                        outptr += 4;
                    }
                }

                if (elempack == 4 && out_features_elempack == 1) {
                    float* outptr = top_blob_ra[j].data();

                    for (int p = 0; p < out_features; p++) {
                        const float* kptr = (const float*)weight_data_tm_ra.data() + in_features * p;
                        const float* m = bottom_blob_ra[j].data();

                        __m128 _sum0 = _mm_set1_ps(0.f);
                        __m128 _sum1 = _mm_set1_ps(0.f);
                        __m128 _sum2 = _mm_set1_ps(0.f);
                        __m128 _sum3 = _mm_set1_ps(0.f);

                        if (bias_term) {
                            _sum0 = _mm_set1_ps(bias_data_ptr[p]);
                        }

                        int i = 0;
                        for (; i + 7 < in_features; i += 8) {
                            __m128 _val0 = _mm_loadu_ps(m);
                            __m128 _val1 = _mm_loadu_ps(m + 4);
                            __m128 _val2 = _mm_loadu_ps(m + 8);
                            __m128 _val3 = _mm_loadu_ps(m + 12);
                            __m128 _val4 = _mm_loadu_ps(m + 16);
                            __m128 _val5 = _mm_loadu_ps(m + 20);
                            __m128 _val6 = _mm_loadu_ps(m + 24);
                            __m128 _val7 = _mm_loadu_ps(m + 28);
                            _sum0 = _mm_add_ps(_mm_mul_ps(_val0, _mm_set1_ps(kptr[0])), _sum0);
                            _sum1 = _mm_add_ps(_mm_mul_ps(_val1, _mm_set1_ps(kptr[1])), _sum1);
                            _sum2 = _mm_add_ps(_mm_mul_ps(_val2, _mm_set1_ps(kptr[2])), _sum2);
                            _sum3 = _mm_add_ps(_mm_mul_ps(_val3, _mm_set1_ps(kptr[3])), _sum3);
                            _sum0 = _mm_add_ps(_mm_mul_ps(_val4, _mm_set1_ps(kptr[4])), _sum0);
                            _sum1 = _mm_add_ps(_mm_mul_ps(_val5, _mm_set1_ps(kptr[5])), _sum1);
                            _sum2 = _mm_add_ps(_mm_mul_ps(_val6, _mm_set1_ps(kptr[6])), _sum2);
                            _sum3 = _mm_add_ps(_mm_mul_ps(_val7, _mm_set1_ps(kptr[7])), _sum3);

                            m += 32;
                            kptr += 8;
                        }
                        for (; i + 3 < in_features; i += 4) {
                            __m128 _val0 = _mm_loadu_ps(m);
                            __m128 _val1 = _mm_loadu_ps(m + 4);
                            __m128 _val2 = _mm_loadu_ps(m + 8);
                            __m128 _val3 = _mm_loadu_ps(m + 12);
                            _sum0 = _mm_add_ps(_mm_mul_ps(_val0, _mm_set1_ps(kptr[0])), _sum0);
                            _sum1 = _mm_add_ps(_mm_mul_ps(_val1, _mm_set1_ps(kptr[1])), _sum1);
                            _sum2 = _mm_add_ps(_mm_mul_ps(_val2, _mm_set1_ps(kptr[2])), _sum2);
                            _sum3 = _mm_add_ps(_mm_mul_ps(_val3, _mm_set1_ps(kptr[3])), _sum3);

                            m += 16;
                            kptr += 4;
                        }
                        for (; i < in_features; i++) {
                            __m128 _val = _mm_loadu_ps(m);
                            __m128 _k = _mm_set1_ps(kptr[0]);
                            _sum0 = _mm_add_ps(_mm_mul_ps(_val, _k), _sum0);

                            m += 4;
                            kptr += 1;
                        }

                        _sum0 = _mm_add_ps(_sum0, _sum1);
                        _sum2 = _mm_add_ps(_sum2, _sum3);
                        _sum0 = _mm_add_ps(_sum0, _sum2);

                        _sum0 = activation_sse(_sum0, activation_type, activation_params);

                        _mm_storeu_ps(outptr, _sum0);
                        outptr += 4;
                    }
                }
    #endif // __SSE2__

                if (elempack == 1 && out_features_elempack == 1) {
                    float* outptr = top_blob_ra[j].data();

                    for (int p = 0; p < out_features; p++) {
                        const float* kptr = (const float*)weight_data_tm_ra.data() + in_features * p;
                        const float* m = bottom_blob_ra[j].data();

                        float sum = 0.f;

                        if (bias_term) {
                            sum = bias_data_ptr[p];
                        }

                        int i = 0;
    #if __SSE2__
    #if __AVX__
                        __m256 _sum = _mm256_set1_ps(0.f);
                        for (; i + 7 < in_features; i += 8) {
                            __m256 _m = _mm256_loadu_ps(m);
                            __m256 _w = _mm256_loadu_ps(kptr);
                            _sum = _mm256_comp_fmadd_ps(_m, _w, _sum);

                            m += 8;
                            kptr += 8;
                        }
    #endif // __AVX__
                        __m128 _suml = _mm_set1_ps(0.f);
                        for (; i + 3 < in_features; i += 4) {
                            __m128 _val = _mm_loadu_ps(m);
                            __m128 _k = _mm_loadu_ps(kptr);
                            _suml = _mm_add_ps(_mm_mul_ps(_val, _k), _suml);

                            m += 4;
                            kptr += 4;
                        }
    #endif // __SSE2__
                        for (; i < in_features; i++) {
                            sum += *m++ * *kptr++;
                        }

    #if __SSE2__
    #if __AVX__
                        sum += _mm256_reduce_add_ps(_sum);
    #endif // __AVX__
                        sum += _mm_reduce_add_ps(_suml);
    #endif // __SSE2__

                        sum = activation_ss(sum, activation_type, activation_params);

                        outptr[0] = sum;
                        outptr += 1;
                    }
                }
            }
        });

        return 0;
    }

    // flatten
    Tensor bottom_blob_flattened = bottom_blob.flatten(0);

    int out_elempack = 1;
#if __SSE2__
    if (opt.use_packing_layout) {
#if __AVX512F__
        out_elempack = out_features % 16 == 0 ? 16 : out_features % 8 == 0 ? 8 : out_features % 4 == 0 ? 4 : 1;
#elif __AVX__
        out_elempack = out_features % 8 == 0 ? 8 : out_features % 4 == 0 ? 4 : 1;
#else
        out_elempack = out_features % 4 == 0 ? 4 : 1;
#endif
    }
#endif // __SSE2__
    
    top_blob = otter::empty({out_features / out_elempack}, otter::get_update_scalarType(otter::ScalarType::Float, out_elempack));
    
    auto top_blob_ra = top_blob.raw_accessor<float, 1>();
    auto weight_data_tm_ra = weight_data_tm.raw_accessor<float, 2>();
    const float* bias_data_ptr = (const float*)bias_data.data_ptr();

#if __SSE2__
#if __AVX__
    if (out_elempack == 8) {
        otter::parallel_for(0, out_features / out_elempack, 0, [&](int64_t begin, int64_t end) {
            for (const auto p : otter::irange(begin, end)) {
                __m256 _sum0 = _mm256_set1_ps(0.f);
                __m256 _sum1 = _mm256_set1_ps(0.f);
                __m256 _sum2 = _mm256_set1_ps(0.f);
                __m256 _sum3 = _mm256_set1_ps(0.f);
                __m256 _sum4 = _mm256_set1_ps(0.f);
                __m256 _sum5 = _mm256_set1_ps(0.f);
                __m256 _sum6 = _mm256_set1_ps(0.f);
                __m256 _sum7 = _mm256_set1_ps(0.f);

                if (bias_term) {
                    _sum0 = _mm256_loadu_ps((const float*)bias_data_ptr + p * 8);
                }

                const float* kptr = weight_data_tm_ra[p].data();

                const float* sptr = (const float*)bottom_blob_flattened.data_ptr();

                int i = 0;
                for (; i + 7 < in_features; i += 8) {
                    __m256 _val0 = _mm256_broadcast_ss(sptr);
                    __m256 _val1 = _mm256_broadcast_ss(sptr + 1);
                    __m256 _val2 = _mm256_broadcast_ss(sptr + 2);
                    __m256 _val3 = _mm256_broadcast_ss(sptr + 3);
                    __m256 _val4 = _mm256_broadcast_ss(sptr + 4);
                    __m256 _val5 = _mm256_broadcast_ss(sptr + 5);
                    __m256 _val6 = _mm256_broadcast_ss(sptr + 6);
                    __m256 _val7 = _mm256_broadcast_ss(sptr + 7);

                    __m256 _w0 = _mm256_loadu_ps(kptr);
                    _sum0 = _mm256_comp_fmadd_ps(_val0, _w0, _sum0);
                    __m256 _w1 = _mm256_loadu_ps(kptr + 8);
                    _sum1 = _mm256_comp_fmadd_ps(_val1, _w1, _sum1);
                    __m256 _w2 = _mm256_loadu_ps(kptr + 16);
                    _sum2 = _mm256_comp_fmadd_ps(_val2, _w2, _sum2);
                    __m256 _w3 = _mm256_loadu_ps(kptr + 24);
                    _sum3 = _mm256_comp_fmadd_ps(_val3, _w3, _sum3);
                    __m256 _w4 = _mm256_loadu_ps(kptr + 32);
                    _sum4 = _mm256_comp_fmadd_ps(_val4, _w4, _sum4);
                    __m256 _w5 = _mm256_loadu_ps(kptr + 40);
                    _sum5 = _mm256_comp_fmadd_ps(_val5, _w5, _sum5);
                    __m256 _w6 = _mm256_loadu_ps(kptr + 48);
                    _sum6 = _mm256_comp_fmadd_ps(_val6, _w6, _sum6);
                    __m256 _w7 = _mm256_loadu_ps(kptr + 56);
                    _sum7 = _mm256_comp_fmadd_ps(_val7, _w7, _sum7);

                    sptr += 8;
                    kptr += 64;
                }
                for (; i + 3 < in_features; i += 4) {
                    __m256 _val0 = _mm256_broadcast_ss(sptr);
                    __m256 _val1 = _mm256_broadcast_ss(sptr + 1);
                    __m256 _val2 = _mm256_broadcast_ss(sptr + 2);
                    __m256 _val3 = _mm256_broadcast_ss(sptr + 3);

                    __m256 _w0 = _mm256_loadu_ps(kptr);
                    _sum0 = _mm256_comp_fmadd_ps(_val0, _w0, _sum0);
                    __m256 _w1 = _mm256_loadu_ps(kptr + 8);
                    _sum1 = _mm256_comp_fmadd_ps(_val1, _w1, _sum1);
                    __m256 _w2 = _mm256_loadu_ps(kptr + 16);
                    _sum2 = _mm256_comp_fmadd_ps(_val2, _w2, _sum2);
                    __m256 _w3 = _mm256_loadu_ps(kptr + 24);
                    _sum3 = _mm256_comp_fmadd_ps(_val3, _w3, _sum3);

                    sptr += 4;
                    kptr += 32;
                }
                for (; i < in_features; i++) {
                    __m256 _val = _mm256_set1_ps(sptr[0]);
                    __m256 _w = _mm256_loadu_ps(kptr);
                    _sum0 = _mm256_comp_fmadd_ps(_val, _w, _sum0);

                    sptr += 1;
                    kptr += 8;
                }

                _sum0 = _mm256_add_ps(_sum0, _sum1);
                _sum2 = _mm256_add_ps(_sum2, _sum3);
                _sum4 = _mm256_add_ps(_sum4, _sum5);
                _sum6 = _mm256_add_ps(_sum6, _sum7);
                _sum0 = _mm256_add_ps(_sum0, _sum2);
                _sum4 = _mm256_add_ps(_sum4, _sum6);
                _sum0 = _mm256_add_ps(_sum0, _sum4);

                _sum0 = activation_avx(_sum0, activation_type, activation_params);

                float* outptr = top_blob_ra.data();
                _mm256_storeu_ps(outptr + p * 8, _sum0);
            }
        });
    }
#endif // __AVX__

    if (out_elempack == 4) {
        otter::parallel_for(0, out_features / out_elempack, 0, [&](int64_t begin, int64_t end) {
            for (const auto p : otter::irange(begin, end)) {
                __m128 _sum0 = _mm_set1_ps(0.f);
                __m128 _sum1 = _mm_set1_ps(0.f);
                __m128 _sum2 = _mm_set1_ps(0.f);
                __m128 _sum3 = _mm_set1_ps(0.f);
    #if __AVX__
                __m128 _sum4 = _mm_set1_ps(0.f);
                __m128 _sum5 = _mm_set1_ps(0.f);
                __m128 _sum6 = _mm_set1_ps(0.f);
                __m128 _sum7 = _mm_set1_ps(0.f);
    #endif

                if (bias_term) {
                    _sum0 = _mm_loadu_ps((const float*)bias_data_ptr + p * 4);
                }

                const float* kptr = weight_data_tm_ra[p].data();

                const float* sptr = (const float*)bottom_blob_flattened.data_ptr();

                int i = 0;
    #if __AVX__
                for (; i + 7 < in_features; i += 8) {
                    __m128 _val0 = _mm_broadcast_ss(sptr);
                    __m128 _val1 = _mm_broadcast_ss(sptr + 1);
                    __m128 _val2 = _mm_broadcast_ss(sptr + 2);
                    __m128 _val3 = _mm_broadcast_ss(sptr + 3);
                    __m128 _val4 = _mm_broadcast_ss(sptr + 4);
                    __m128 _val5 = _mm_broadcast_ss(sptr + 5);
                    __m128 _val6 = _mm_broadcast_ss(sptr + 6);
                    __m128 _val7 = _mm_broadcast_ss(sptr + 7);

                    __m128 _w0 = _mm_loadu_ps(kptr);
                    _sum0 = _mm_comp_fmadd_ps(_val0, _w0, _sum0);
                    __m128 _w1 = _mm_loadu_ps(kptr + 4);
                    _sum1 = _mm_comp_fmadd_ps(_val1, _w1, _sum1);
                    __m128 _w2 = _mm_loadu_ps(kptr + 8);
                    _sum2 = _mm_comp_fmadd_ps(_val2, _w2, _sum2);
                    __m128 _w3 = _mm_loadu_ps(kptr + 12);
                    _sum3 = _mm_comp_fmadd_ps(_val3, _w3, _sum3);
                    __m128 _w4 = _mm_loadu_ps(kptr + 16);
                    _sum4 = _mm_comp_fmadd_ps(_val4, _w4, _sum4);
                    __m128 _w5 = _mm_loadu_ps(kptr + 20);
                    _sum5 = _mm_comp_fmadd_ps(_val5, _w5, _sum5);
                    __m128 _w6 = _mm_loadu_ps(kptr + 24);
                    _sum6 = _mm_comp_fmadd_ps(_val6, _w6, _sum6);
                    __m128 _w7 = _mm_loadu_ps(kptr + 28);
                    _sum7 = _mm_comp_fmadd_ps(_val7, _w7, _sum7);

                    sptr += 8;
                    kptr += 32;
                }
    #endif
                for (; i + 3 < in_features; i += 4) {
                    __m128 _val0 = _mm_set1_ps(sptr[0]);
                    __m128 _val1 = _mm_set1_ps(sptr[1]);
                    __m128 _val2 = _mm_set1_ps(sptr[2]);
                    __m128 _val3 = _mm_set1_ps(sptr[3]);

                    __m128 _w0 = _mm_loadu_ps(kptr);
                    _sum0 = _mm_add_ps(_mm_mul_ps(_val0, _w0), _sum0);
                    __m128 _w1 = _mm_loadu_ps(kptr + 4);
                    _sum1 = _mm_add_ps(_mm_mul_ps(_val1, _w1), _sum1);
                    __m128 _w2 = _mm_loadu_ps(kptr + 8);
                    _sum2 = _mm_add_ps(_mm_mul_ps(_val2, _w2), _sum2);
                    __m128 _w3 = _mm_loadu_ps(kptr + 12);
                    _sum3 = _mm_add_ps(_mm_mul_ps(_val3, _w3), _sum3);

                    sptr += 4;
                    kptr += 16;
                }
                for (; i < in_features; i++) {
                    __m128 _val = _mm_set1_ps(sptr[0]);
                    __m128 _w = _mm_loadu_ps(kptr);
                    _sum0 = _mm_add_ps(_mm_mul_ps(_val, _w), _sum0);

                    sptr += 1;
                    kptr += 4;
                }

                _sum0 = _mm_add_ps(_sum0, _sum1);
                _sum2 = _mm_add_ps(_sum2, _sum3);
    #if __AVX__
                _sum4 = _mm_add_ps(_sum4, _sum5);
                _sum6 = _mm_add_ps(_sum6, _sum7);
    #endif
                _sum0 = _mm_add_ps(_sum0, _sum2);
    #if __AVX__
                _sum4 = _mm_add_ps(_sum4, _sum6);
                _sum0 = _mm_add_ps(_sum0, _sum4);
    #endif

                _sum0 = activation_sse(_sum0, activation_type, activation_params);

                float* outptr = top_blob_ra.data();
                _mm_storeu_ps(outptr + p * 4, _sum0);
            }
        });
    }
#endif // __SSE2__

    if (out_elempack == 1)
    {
#if __SSE2__
#if __AVX__
        int remain_out_features_start = 0;
        int nn_out_features = out_features >> 3;

        otter::parallel_for(0, nn_out_features, 0, [&](int64_t begin, int64_t end) {
            for (const auto pp : otter::irange(begin, end)) {
                int p = pp * 8;

                float sums[8] = {0.0f};
                if (bias_term) {
                    sums[0] = bias_data_ptr[p];
                    sums[1] = bias_data_ptr[p + 1];
                    sums[2] = bias_data_ptr[p + 2];
                    sums[3] = bias_data_ptr[p + 3];
                    sums[4] = bias_data_ptr[p + 4];
                    sums[5] = bias_data_ptr[p + 5];
                    sums[6] = bias_data_ptr[p + 6];
                    sums[7] = bias_data_ptr[p + 7];
                }

                const float* w0 = (const float*)weight_data_tm_ra.data() + in_features * p;
                const float* w1 = (const float*)weight_data_tm_ra.data() + in_features * (p + 1);
                const float* w2 = (const float*)weight_data_tm_ra.data() + in_features * (p + 2);
                const float* w3 = (const float*)weight_data_tm_ra.data() + in_features * (p + 3);
                const float* w4 = (const float*)weight_data_tm_ra.data() + in_features * (p + 4);
                const float* w5 = (const float*)weight_data_tm_ra.data() + in_features * (p + 5);
                const float* w6 = (const float*)weight_data_tm_ra.data() + in_features * (p + 6);
                const float* w7 = (const float*)weight_data_tm_ra.data() + in_features * (p + 7);

                const float* m = (const float*)bottom_blob_flattened.data_ptr();

                __m256 _sum0 = _mm256_set1_ps(0.f);
                __m256 _sum1 = _mm256_set1_ps(0.f);
                __m256 _sum2 = _mm256_set1_ps(0.f);
                __m256 _sum3 = _mm256_set1_ps(0.f);
                __m256 _sum4 = _mm256_set1_ps(0.f);
                __m256 _sum5 = _mm256_set1_ps(0.f);
                __m256 _sum6 = _mm256_set1_ps(0.f);
                __m256 _sum7 = _mm256_set1_ps(0.f);

                int i = 0;
                for (; i + 7 < in_features; i += 8) {
                    __m256 _m = _mm256_loadu_ps(m);

                    __m256 _w0 = _mm256_loadu_ps(w0);
                    _sum0 = _mm256_comp_fmadd_ps(_m, _w0, _sum0);
                    __m256 _w1 = _mm256_loadu_ps(w1);
                    _sum1 = _mm256_comp_fmadd_ps(_m, _w1, _sum1);
                    __m256 _w2 = _mm256_loadu_ps(w2);
                    _sum2 = _mm256_comp_fmadd_ps(_m, _w2, _sum2);
                    __m256 _w3 = _mm256_loadu_ps(w3);
                    _sum3 = _mm256_comp_fmadd_ps(_m, _w3, _sum3);
                    __m256 _w4 = _mm256_loadu_ps(w4);
                    _sum4 = _mm256_comp_fmadd_ps(_m, _w4, _sum4);
                    __m256 _w5 = _mm256_loadu_ps(w5);
                    _sum5 = _mm256_comp_fmadd_ps(_m, _w5, _sum5);
                    __m256 _w6 = _mm256_loadu_ps(w6);
                    _sum6 = _mm256_comp_fmadd_ps(_m, _w6, _sum6);
                    __m256 _w7 = _mm256_loadu_ps(w7);
                    _sum7 = _mm256_comp_fmadd_ps(_m, _w7, _sum7);

                    m += 8;
                    w0 += 8;
                    w1 += 8;
                    w2 += 8;
                    w3 += 8;
                    w4 += 8;
                    w5 += 8;
                    w6 += 8;
                    w7 += 8;
                }
                for (; i < in_features; i++) {
                    sums[0] += *m * *w0;
                    sums[1] += *m * *w1;
                    sums[2] += *m * *w2;
                    sums[3] += *m * *w3;
                    sums[4] += *m * *w4;
                    sums[5] += *m * *w5;
                    sums[6] += *m * *w6;
                    sums[7] += *m * *w7;

                    m++;
                    w0++;
                    w1++;
                    w2++;
                    w3++;
                    w4++;
                    w5++;
                    w6++;
                    w7++;
                }

                __m256 _sums = HorizontalSums(_sum0, _sum1, _sum2, _sum3, _sum4, _sum5, _sum6, _sum7);
                __m256 _sums_f = _mm256_loadu_ps(sums);
                _sums = _mm256_add_ps(_sums_f, _sums);
                _sums = activation_avx(_sums, activation_type, activation_params);

                float* outptr = top_blob_ra.data();
                _mm256_storeu_ps(outptr + p, _sums);
            }
        });

        remain_out_features_start += (nn_out_features << 3);
        nn_out_features = (out_features - remain_out_features_start) >> 2;
#else
        int remain_out_features_start = 0;
        int nn_out_features = out_features >> 2;
#endif // __AVX__

        otter::parallel_for(0, nn_out_features, 0, [&](int64_t begin, int64_t end) {
            for (const auto pp : otter::irange(begin, end)) {
                int p = remain_out_features_start + (pp * 4);

                float sums[4] = {0.0f};
                if (bias_term) {
                    sums[0] = bias_data_ptr[p];
                    sums[1] = bias_data_ptr[p + 1];
                    sums[2] = bias_data_ptr[p + 2];
                    sums[3] = bias_data_ptr[p + 3];
                }

                const float* w0 = (const float*)weight_data_tm_ra.data() + in_features * p;
                const float* w1 = (const float*)weight_data_tm_ra.data() + in_features * (p + 1);
                const float* w2 = (const float*)weight_data_tm_ra.data() + in_features * (p + 2);
                const float* w3 = (const float*)weight_data_tm_ra.data() + in_features * (p + 3);

                const float* m = (const float*)bottom_blob_flattened.data_ptr();

                int i = 0;
    #if __AVX__
                __m256 _sum0 = _mm256_set1_ps(0.f);
                __m256 _sum1 = _mm256_set1_ps(0.f);
                __m256 _sum2 = _mm256_set1_ps(0.f);
                __m256 _sum3 = _mm256_set1_ps(0.f);
                for (; i + 7 < in_features; i += 8) {
                    __m256 _m = _mm256_loadu_ps(m);

                    __m256 _w0 = _mm256_loadu_ps(w0);
                    _sum0 = _mm256_comp_fmadd_ps(_m, _w0, _sum0);
                    __m256 _w1 = _mm256_loadu_ps(w1);
                    _sum1 = _mm256_comp_fmadd_ps(_m, _w1, _sum1);
                    __m256 _w2 = _mm256_loadu_ps(w2);
                    _sum2 = _mm256_comp_fmadd_ps(_m, _w2, _sum2);
                    __m256 _w3 = _mm256_loadu_ps(w3);
                    _sum3 = _mm256_comp_fmadd_ps(_m, _w3, _sum3);

                    m += 8;
                    w0 += 8;
                    w1 += 8;
                    w2 += 8;
                    w3 += 8;
                }
    #endif // __AVX__
                __m128 _sum0l = _mm_set1_ps(0.f);
                __m128 _sum1l = _mm_set1_ps(0.f);
                __m128 _sum2l = _mm_set1_ps(0.f);
                __m128 _sum3l = _mm_set1_ps(0.f);
                for (; i + 3 < in_features; i += 4) {
                    __m128 _m = _mm_loadu_ps(m);

                    __m128 _w0 = _mm_loadu_ps(w0);
                    _sum0l = _mm_add_ps(_mm_mul_ps(_m, _w0), _sum0l);
                    __m128 _w1 = _mm_loadu_ps(w1);
                    _sum1l = _mm_add_ps(_mm_mul_ps(_m, _w1), _sum1l);
                    __m128 _w2 = _mm_loadu_ps(w2);
                    _sum2l = _mm_add_ps(_mm_mul_ps(_m, _w2), _sum2l);
                    __m128 _w3 = _mm_loadu_ps(w3);
                    _sum3l = _mm_add_ps(_mm_mul_ps(_m, _w3), _sum3l);

                    m += 4;
                    w0 += 4;
                    w1 += 4;
                    w2 += 4;
                    w3 += 4;
                }
                for (; i < in_features; i++) {
                    sums[0] += *m * *w0;
                    sums[1] += *m * *w1;
                    sums[2] += *m * *w2;
                    sums[3] += *m * *w3;

                    m++;
                    w0++;
                    w1++;
                    w2++;
                    w3++;
                }

                __m128 _sums = _mm_loadu_ps(sums);
    #if __AVX__
                _sums = _mm_add_ps(HorizontalSums(_sum0, _sum1, _sum2, _sum3), _sums);
    #endif
                _MM_TRANSPOSE4_PS(_sum0l, _sum1l, _sum2l, _sum3l);
                _sums = _mm_add_ps(_sum0l, _sums);
                _sums = _mm_add_ps(_sum1l, _sums);
                _sums = _mm_add_ps(_sum2l, _sums);
                _sums = _mm_add_ps(_sum3l, _sums);
                _sums = activation_sse(_sums, activation_type, activation_params);

                float* outptr = top_blob_ra.data();
                _mm_storeu_ps(outptr + p, _sums);
            }
        });

        remain_out_features_start += (nn_out_features << 2);
#else
        int remain_out_features_start = 0;
#endif // __SSE2__

        otter::parallel_for(remain_out_features_start, out_features, 0, [&](int64_t begin, int64_t end) {
            for (const auto p : otter::irange(begin, end)) {
                float sum = 0.f;

                if (bias_term)
                    sum = bias_data_ptr[p];

                const float* w = (const float*)weight_data_tm_ra.data() + in_features * p;

                const float* m = (const float*)bottom_blob_flattened.data_ptr();

                int i = 0;
    #if __SSE2__
    #if __AVX__
                __m256 _sum = _mm256_set1_ps(0.f);
                for (; i + 7 < in_features; i += 8) {
                    __m256 _m = _mm256_loadu_ps(m);

                    __m256 _w = _mm256_loadu_ps(w);
                    _sum = _mm256_comp_fmadd_ps(_m, _w, _sum);

                    m += 8;
                    w += 8;
                }
    #endif // __AVX__
                __m128 _suml = _mm_set1_ps(0.f);
                for (; i + 3 < in_features; i += 4) {
                    __m128 _m = _mm_loadu_ps(m);

                    __m128 _w = _mm_loadu_ps(w);
                    _suml = _mm_add_ps(_mm_mul_ps(_m, _w), _suml);

                    m += 4;
                    w += 4;
                }
    #endif // __SSE2__
                for (; i < in_features; i++) {
                    sum += *m * *w;
                    m++;
                    w++;
                }

    #if __SSE2__
    #if __AVX__
                sum += _mm256_reduce_add_ps(_sum);
    #endif
                sum += _mm_reduce_add_ps(_suml);
    #endif // __SSE2__

                sum = activation_ss(sum, activation_type, activation_params);

                float* outptr = top_blob_ra.data();
                outptr[p] = sum;
            }
        });
    }

    return 0;
}

//int InnerProductLayer::forward(const Tensor& bottom_blob, Tensor& top_blob, const NetOption& opt) const {
//
//    top_blob = otter::empty({out_features}, otter::ScalarType::Float);
//
//    const float* weight_data_ptr = weight_data.data_ptr<float>();
//    const float* bias_data_ptr = (bias_term) ? bias_data.data_ptr<float>() : nullptr;
//
//    if (bottom_blob.dim() == 2 && bottom_blob.size(1); == in_features && h > 1) {
//        int w = bottom_blob.size(1);
//        int h = bottom_blob.size(0);
//
//        // gemm
//        top_blob = otter::empty({h, out_features}, otter::ScalarType::Float);
//
//        auto bottom_blob_a = bottom_blob.accessor<float, 2>();
//        auto top_blob_a = top_blob.accessor<float, 2>();
//
//        otter::parallel_for(0, h, 0, [&](int64_t begin, int64_t end) {
//            for (const auto j : otter::irange(begin, end)) {
//                const float* m = bottom_blob_a[j].data();
//                float* outptr = top_blob_a[j].data();
//
//                for (int p = 0; p < out_features; p++) {
//                    const float* kptr = (const float*)weight_data_ptr + w * p;
//
//                    float sum = 0.f;
//
//                    if (bias_term)
//                        sum = bias_data_ptr[p];
//
//                    for (int i = 0; i < w; i++) {
//                        sum += m[i] * kptr[i];
//                    }
//
//                    outptr[p] = activation_ss(sum, activation_type, activation_params);
//                }
//            }
//        });
//
//        return 0;
//    }
//
//    int channels = 1;   // TODO: temp
//    int size = bottom_blob.size(0); // TODO: temp
//    auto bottom_blob_ra = bottom_blob.accessor<float, 1>();
//
//    otter::parallel_for(0, out_features, 0, [&](int64_t begin, int64_t end) {
//        for (const auto p : otter::irange(begin, end)) {
//            float sum = 0.f;
//
//            if (bias_term)
//                sum = bias_data_ptr[p];
//
//            // channels
//            for (int q = 0; q < channels; q++) {
//                const float* w = (const float*)weight_data_ptr + size * channels * p + size * q;
//                const float* m = &bottom_blob_ra[q];
//
//                for (int i = 0; i < size; i++) {
//                    sum += m[i] * w[i];
//                }
//            }
//
//            top_blob[p] = activation_ss(sum, activation_type, activation_params);
//        }
//    });
//
//    return 0;
//}


}   // end namespace otter
