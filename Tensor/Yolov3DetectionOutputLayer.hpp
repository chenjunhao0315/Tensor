//
//  Yolov3DetectionOutputLayer.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/3/6.
//

#ifndef Yolov3DetectionOutputLayer_hpp
#define Yolov3DetectionOutputLayer_hpp

#include "Layer.hpp"

namespace otter {

class Yolov3DetectionOutputLayer : public Layer {
public:
    Yolov3DetectionOutputLayer();
    
    virtual int parse_param(LayerOption& option, ParamDict& pd);
    
    virtual int compute_output_shape(ParamDict& pd);
    
    virtual int load_param(const ParamDict &pd);
    
    virtual int forward(const std::vector<Tensor>& bottom_blobs, std::vector<Tensor>& top_blobs, const NetOption& opt) const;
    
    virtual std::string type() const { return "Yolov3"; }
    
public:
    int num_class;
    int num_box;
    float confidence_threshold;
    float nms_threshold;
    
    float scale_x_y;
    
    Tensor biases;
    Tensor mask;
    Tensor anchors_scale;
    
public:
    struct BBox {
        int label;
        float score;
        float xmin;
        float ymin;
        float xmax;
        float ymax;
        float area;
    };
    
    void qsort_descent_inplace(std::vector<BBox>& datas, int left, int right) const;
    void qsort_descent_inplace(std::vector<BBox>& datas) const;
    void nms_sorted_bboxes(std::vector<BBox>& bboxes, std::vector<size_t>& picked, float nms_threshold) const;
};

enum class Yolov3DetectionParam {
    Num_class,
    Num_box,
    Confidence_threshold,
    Nms_threshold,
    Biases,
    Mask,
    Anchors_scale,
    Scale_x_y,
    Input_height,
    Input_width
};

}

#endif /* Yolov3DetectionOutputLayer_hpp */
