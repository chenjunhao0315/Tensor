//
//  Net.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/2/19.
//

#include "Exception.hpp"

#include "Net.hpp"
#include "LayerRegistry.hpp"

#include "Formatting.hpp"

namespace otter {

Net::Net() {
    
}

Net::~Net() {
    
}

void Net::init_blobs_and_layers(size_t blob_count, size_t layer_count) {
    blobs.resize((size_t)blob_count);
    layers.resize((size_t)layer_count);
}

void Net::addLayer(LayerOption option) {
    std::string type = opt_find_string(option, "type", "Undefined");
    
    // Check Relu bias
    
    if (!opt_check_string(option, "name")) {
        option["name"] = std::to_string(layer_options.size());
    }
    
    if (!opt_check_string(option, "input_id")) {
        option["input_id"] = "0";
    }
    
    blob_count_++;   // TODO: Check Split layer or etc. which will produce multi blobs
    layer_options.push_back(option);
    
    if (opt_check_string(option, "batchnorm")) {
        LayerOption auto_option;
        auto_option["type"] = "BatchNormalization";
        auto_option["name"] = "bn_" + option["name"];
        blob_count_++;
        layer_options.push_back(auto_option);
    }
    
    if (opt_check_string(option, "activation")) {
        LayerOption auto_option;
        std::string activation = option["activation"];
        auto_option["type"] = activation;
        std::string abbreviate = activation.substr(0, 2);
        std::transform(abbreviate.begin(), abbreviate.end(), abbreviate.begin(),
            [](unsigned char c){ return std::tolower(c); });
        auto_option["name"] = abbreviate + "_" + option["name"];
        blob_count_++;
        layer_options.push_back(auto_option);
    }
}

void Net::compile() {
    
    size_t layer_count = layer_options.size();
    size_t blob_count  = blob_count_;
    
    OTTER_CHECK(!(layer_count <= 0 || blob_count <= 0), "Invalid network\n");
    
    this->init_blobs_and_layers(blob_count, layer_count);
    
    ParamDict pd;
    
    int blob_index = 0;
    for (const auto i : otter::irange(layer_count)) {
        LayerOption& option = layer_options[i];
        
        std::string type = option["type"];
        std::string name = option["name"];
        int bottom_count = (type == "Input") ? 0 : (int)std::count(option["input"].begin(), option["input"].end(), ',') + 1;
        int top_count    = (int)std::count(option["output"].begin(), option["output"].end(), ',') + 1;
        
        Layer* layer = LayerRegistry::CreateLayer(type);
        layer->name = name;
        
        printf("Create layer %d Type: %s Name: %s\n", (int)i, layer->type().c_str(), name.c_str());
        
        layer->bottoms.resize(bottom_count);
        std::stringstream bottom_list(option["input"]);
        for (const auto j : otter::irange(bottom_count)) {
            std::string bottom_name;
            std::getline(bottom_list, bottom_name, ',');
            EARSE_SPACE(bottom_name);
            
            int bottom_blob_index = find_blob_index_by_name(bottom_name);
            if (bottom_blob_index == -1) {
                Blob& blob = blobs[blob_index];
                bottom_blob_index = blob_index;
                blob.name = std::string(bottom_name);
                
                printf("New Blob index: %d name: %s\n", blob_index, bottom_name.c_str());
                blob_index++;
            }
            
            Blob& blob = blobs[bottom_blob_index];
            blob.consumer = (int)i;
            layer->bottoms[j] = bottom_blob_index;
        }
        
        layer->tops.resize(top_count);
        std::stringstream blob_list(option["output"]);
        for (const auto j : otter::irange(top_count)) {
            std::string blob_name;
            std::getline(blob_list, blob_name, ',');
            EARSE_SPACE(blob_name);
            
            Blob& blob = blobs[blob_index];
            blob.name = blob_name;
            printf("New Blob name: %s\n", blob_name.c_str());

            blob.producer = (int)i;
            layer->tops[j] = blob_index;

            blob_index++;
        }
        
        int pd_state = layer->prase_param(option, pd);
        if (pd_state != 0) {
            fprintf(stderr, "ParamDict load %s failed or undefined\n", name.c_str());
            continue;
        }
        
        Tensor shape_hints = pd.get(30, Tensor());
        if (shape_hints.defined()) {
            for (const auto j : otter::irange(top_count)) {
                Blob& blob = blobs[layer->tops[j]];
                
                blob.shape = shape_hints.clone();
            }
        }
        
        layer->bottom_shapes.resize(bottom_count);
        for (int j = 0; j < bottom_count; j++) {
            layer->bottom_shapes[j] = blobs[layer->bottoms[j]].shape;
        }

        layer->top_shapes.resize(top_count);
        for (int j = 0; j < top_count; j++) {
            layer->top_shapes[j] = blobs[layer->tops[j]].shape;
        }
        
        int load_state = layer->load_param(pd);
        if (load_state != 0) {
            fprintf(stderr, "Layer load %lu %s failed or undefined\n", i, layer->name.c_str());
            continue;
        }
        
        layers[i] = layer;
    }
    
    this->update_input_output_indexes();
    this->update_input_output_names();
    
    // Debug
    for (int i = 0; i < blobs.size(); ++i) {
        printf("Blob %d name: %s producer: %d consumer: %d shape: ", i, blobs[i].name.c_str(), blobs[i].producer, blobs[i].consumer);
        std::cout << blobs[i].shape << std::endl;
    }
}

int Net::find_blob_index_by_name(std::string name) const {
    for (const auto i : otter::irange(blobs.size())) {
        const Blob& blob = blobs[i];
        if (blob.name == name) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

void Net::update_input_output_indexes() {
    input_blob_indexes.clear();
    output_blob_indexes.clear();

    for (size_t i = 0; i < layers.size(); i++) {
        if (layers[i]->type() == "Input") {
            int blob_index = layers[i]->tops[0];
            input_blob_indexes.push_back(blob_index);
        }
    }

    for (size_t i = 0; i < blobs.size(); i++) {
        if (blobs[i].producer != -1 && blobs[i].consumer == -1) {
            output_blob_indexes.push_back(i);
        }
    }
}

void Net::update_input_output_names()
{
    input_blob_names.clear();
    output_blob_names.clear();

    for (size_t i = 0; i < input_blob_indexes.size(); i++)
    {
        int blob_index = input_blob_indexes[i];
        input_blob_names.push_back(blobs[blob_index].name.c_str());
    }

    for (size_t i = 0; i < output_blob_indexes.size(); i++)
    {
        int blob_index = output_blob_indexes[i];
        output_blob_names.push_back(blobs[blob_index].name.c_str());
    }
}

int Net::forward_layer(int layer_index, std::vector<Tensor>& blob_tensors, const NetOption& opt) const {
    const Layer* layer = layers[layer_index];
    
    if (layer->one_blob_only) {
        int bottom_blob_index = layer->bottoms[0];
        
        if (!blob_tensors[bottom_blob_index].defined()) {
            int ret = forward_layer(blobs[bottom_blob_index].producer, blob_tensors, opt);
            if (ret != 0)
                return ret;
        }
    } else {
        for (const auto i : otter::irange(layer->bottoms.size())) {
            int bottom_blob_index = layer->bottoms[i];

            if (!blob_tensors[bottom_blob_index].defined()) {
                int ret = forward_layer(blobs[bottom_blob_index].producer, blob_tensors, opt);
                if (ret != 0)
                    return ret;
            }
        }
    }
    
    int ret = do_forward_layer(layer, blob_tensors, opt);
    if (ret != 0)
            return ret;
    
    return 0;
}

int Net::do_forward_layer(const Layer* layer, std::vector<Tensor>& blob_tensors, const NetOption& opt) const {
    if (layer->one_blob_only) {
        int bottom_blob_index = layer->bottoms[0];
        int top_blob_index = layer->tops[0];

        Tensor& bottom_blob_ref = blob_tensors[bottom_blob_index];
        Tensor bottom_blob;
        
        if (opt.lightmode) {
            if (layer->support_inplace && bottom_blob_ref.use_count() != 1) {
                bottom_blob = bottom_blob_ref.clone();
            }
        }
        if (!bottom_blob.defined()) {
            bottom_blob = bottom_blob_ref;
        }
        
        // TODO: Support quantinzation
        // convert_layout(bottom_blob, layer, opt);
        
        if (opt.lightmode && layer->support_inplace) {
            Tensor& bottom_top_blob = bottom_blob;
            int ret = layer->forward_inplace(bottom_top_blob, opt);
            if (ret != 0)
                return ret;

            // store top blob
            blob_tensors[top_blob_index] = bottom_top_blob;
        } else {
            Tensor top_blob;
            int ret = layer->forward(bottom_blob, top_blob, opt);
            if (ret != 0)
                return ret;
            
            blob_tensors[top_blob_index] = top_blob;
        }
        
        if (opt.lightmode) {
            blob_tensors[bottom_blob_index].reset();
        }
    } else {
        std::vector<Tensor> bottom_blobs(layer->bottoms.size());
        for (const auto i : otter::irange(layer->bottoms.size())) {
            int bottom_blob_index = layer->bottoms[i];
            Tensor& bottom_blob_ref = blob_tensors[bottom_blob_index];
            bottom_blobs[i].reset();
            
            if (opt.lightmode) {
                if (layer->support_inplace && bottom_blob_ref.use_count() != 1) {
                    bottom_blobs[i] = bottom_blob_ref.clone();
                }
            }
            if (!bottom_blobs[i].defined()) {
                bottom_blobs[i] = bottom_blob_ref;
            }
            
            // TODO: Support quantinzation
            // convert_layout(bottom_blobs[i], layer, opt);
        }
        
        if (opt.lightmode && layer->support_inplace) {
            std::vector<Tensor>& bottom_top_blobs = bottom_blobs;
            int ret = layer->forward_inplace(bottom_top_blobs, opt);
            if (ret != 0)
                return ret;
            
            for (const auto i : otter::irange(layer->tops.size())) {
                int top_blob_index = layer->tops[i];
                
                blob_tensors[top_blob_index] = bottom_top_blobs[i];
            }
        } else {
            std::vector<Tensor> top_blobs(layer->tops.size());
            int ret = layer->forward(bottom_blobs, top_blobs, opt);
            if (ret != 0)
                return ret;
            
            for (const auto i : otter::irange(layer->tops.size())) {
                int top_blob_index = layer->tops[i];
                
                blob_tensors[top_blob_index] = top_blobs[i];
            }
            
        }
        
        if (opt.lightmode) {
            for (const auto i : otter::irange(layer->bottoms.size())) {
                int bottom_blob_index = layer->bottoms[i];
                blob_tensors[bottom_blob_index].reset();
            }
        }
    }
    
    return 0;
}

Extractor Net::create_extractor() const {
    return Extractor(this, blobs.size());
}

Extractor::Extractor(const Net* net, size_t blob_count) {
    net_ = net;
    blob_tensors_.resize(blob_count);
}

void Extractor::clear() {
    blob_tensors_.clear();
}

void Extractor::set_lightmode(bool lightmode) {
    option.lightmode = lightmode;
}

int Extractor::input(std::string blob_name, const Tensor &in) {
    int blob_index = net_->find_blob_index_by_name(blob_name);
    if (blob_index == -1) {
        fprintf(stderr, "Input failed!\n");
    }
    
    return input(blob_index, in);
}

int Extractor::input(int blob_index, const Tensor &in) {
    if (blob_index < 0 ||  blob_index >= (int)blob_tensors_.size())
        return -1;
    
    blob_tensors_[blob_index] = in;
    
    return 0;
}

int Extractor::extract(std::string blob_name, Tensor &feat, int type) {
    int blob_index = net_->find_blob_index_by_name(blob_name);
    if (blob_index == -1) {
        fprintf(stderr, "Extract failed!\n");
    }
    
    return extract(blob_index, feat, type);
}

int Extractor::extract(int blob_index, Tensor &feat, int type) {
    if (blob_index < 0 ||  blob_index >= (int)blob_tensors_.size())
        return -1;
    
    int ret = 0;
    
    if (!blob_tensors_[blob_index].defined()) {
        int layer_index = net_->blobs[blob_index].producer;
        ret = net_->forward_layer(layer_index, blob_tensors_, option);
    }
    
    feat = blob_tensors_[blob_index];
    
    return ret;
}

}
