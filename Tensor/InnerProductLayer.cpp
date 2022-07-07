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

#include "ActivationLayer.hpp"

namespace otter {

InnerProductLayer::InnerProductLayer() {
    one_blob_only = true;
    support_inplace = false;
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
    
    return 0;
}

int InnerProductLayer::forward(const Tensor& bottom_blob, Tensor& top_blob, const NetOption& opt) const {
    
    top_blob = otter::empty({out_features}, otter::ScalarType::Float);
    
    const float* weight_data_ptr = weight_data.data_ptr<float>();
    const float* bias_data_ptr = (bias_term) ? bias_data.data_ptr<float>() : nullptr;
    
    int channels = 1;   // TODO: temp
    int size = bottom_blob.size(0); // TODO: temp
    auto bottom_blob_ra = bottom_blob.accessor<float, 1>();
    
    otter::parallel_for(0, out_features, 0, [&](int64_t begin, int64_t end) {
        for (const auto p : otter::irange(begin, end)) {
            float sum = 0.f;

            if (bias_term)
                sum = bias_data_ptr[p];

            // channels
            for (int q = 0; q < channels; q++) {
                const float* w = (const float*)weight_data_ptr + size * channels * p + size * q;
                const float* m = &bottom_blob_ra[q];

                for (int i = 0; i < size; i++) {
                    sum += m[i] * w[i];
                }
            }

            top_blob[p] = activation_ss(sum, activation_type, activation_params);
        }
    });
    
    return 0;
}


}   // end namespace otter
