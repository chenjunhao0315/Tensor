//
//  NanodetPlusDetectionOutputLayer.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/4/5.
//

#ifndef NanodetPlusDetectionOutputLayer_hpp
#define NanodetPlusDetectionOutputLayer_hpp

#include "Layer.hpp"

namespace otter {

class NanodetPlusDetectionOutputLayer : public Layer {
public:
    NanodetPlusDetectionOutputLayer();
    
    virtual int parse_param(LayerOption& option, ParamDict& pd);
    
    virtual int compute_output_shape(ParamDict& pd);
    
    virtual int load_param(const ParamDict &pd);
    
    virtual int forward(const std::vector<Tensor>& bottom_blobs, std::vector<Tensor>& top_blobs, const NetOption& opt) const;
    
    virtual std::string type() const { return "NanodetPlus"; }
    
public:
    float prob_threshold;
    float nms_threshold;
    otter::Tensor stride;
    
public:
    struct Object {
        float x, y, width, height;
        int label;
        float prob;
    };
    
    void qsort_descent_inplace(std::vector<Object>& faceobjects, int left, int right) const;
    void qsort_descent_inplace(std::vector<Object>& faceobjects) const;
    void nms_sorted_bboxes(const std::vector<Object>& faceobjects, std::vector<int>& picked, float nms_threshold) const;
    void generate_proposals(const otter::Tensor& pred, int stride, float prob_threshold, std::vector<Object>& objects) const;
};

enum class NanodetPlusParam {
    Prob_threshold,
    Nms_threshold,
    Stride
};

Tensor nanodet_pre_process(const Tensor& img, int target_size, float& scale, int& wpad, int& hpad);
Tensor nanodet_post_process(const Tensor& pred, int image_width, int image_height, float scale, int wpad, int hpad);

}   // end namespace otter

#endif /* NanodetPlusDetectionOutputLayer_hpp */
