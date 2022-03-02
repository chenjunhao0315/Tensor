//
//  Layer.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/2/19.
//

#ifndef Layer_hpp
#define Layer_hpp

#include "Tensor.hpp"

#include "ParamDict.hpp"

#include <unordered_map>

namespace otter {

using LayerOption = std::unordered_map<std::string, std::string>;

class NetOption;

class Layer {
public:
    // Initialization
    Layer();
    // Virtual destructor
    virtual ~Layer();
    
    virtual int parse_param(LayerOption& option, ParamDict& pd);
    
    virtual int compute_output_shape(ParamDict& pd);
    
    virtual int load_param(const ParamDict& pd);
    
    virtual int init_model();
    virtual int load_model();
    
    virtual std::string type() const { return "Undefined"; }
    
public:
    // Return 0 for success, else fail
    virtual int forward(const Tensor& bottom_blob, Tensor& top_blob, const NetOption& opt) const;
    virtual int forward(const std::vector<Tensor>& bottom_blobs, std::vector<Tensor>& top_blobs, const NetOption& opt) const;
    
    virtual int forward_inplace(Tensor& bottom_blob, const NetOption& opt) const;
    virtual int forward_inplace(std::vector<Tensor>& bottom_blobs, const NetOption& opt) const;
    
public:
    bool support_inplace;
    bool one_blob_only;
    
public:
    std::vector<int> bottoms;
    std::vector<int> tops;
    std::vector<Tensor> bottom_shapes;
    std::vector<Tensor> top_shapes;
    std::string name;
};

#define OUTPUT_SHAPE_HINT 30

#define opt_find(opt, type) \
    (opt.find(type) != opt.end())
#define opt_get_int(opt, type) \
    int(atoi(opt[type].c_str()))
#define opt_get_float(opt, type) \
    float(atof(opt[type].c_str()))
#define opt_find_int(opt, type, default) \
    (opt.find(type) == opt.end()) ? default : opt_get_int(opt, type)
#define opt_find_float(opt, type, default) \
    (opt.find(type) == opt.end()) ? default : opt_get_float(opt, type)
#define opt_find_string(opt, type, default) \
    (opt.find(type) == opt.end()) ? default : opt[type];
#define opt_check_string(opt, type) \
    (opt.find(type) != opt.end())

}   // end namespace otter

#endif /* Layer_hpp */
