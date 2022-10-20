//
//  Composer.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/10/7.
//

#ifndef Composer_hpp
#define Composer_hpp

#include <vector>

#include "Tensor.hpp"
#include "Net.hpp"
#include "Stabilizer.hpp"
#include "PoseStabilizer.hpp"
#include "Observer.hpp"
#include "DrawDetection.hpp"

namespace otter {
namespace cv {

enum class SelectMethod {
    MAX_AREA,
    MAX_AREA_WITH_ID
};

class Selector {
public:
    Selector() : id_selected(-1) {}
    
    int get_target_index_with_label(Tensor& objects, SelectMethod method, int label);
    void set_id_selected(int id);
private:
    int id_selected;
};

struct Movement {
    otter::cv::Vec2f vec;
    int pitch;
};

class Suggester {
public:
    Suggester() : num_method(2) {}
    
    Movement suggest_portrait(Tensor object, Tensor& keypoints);
private:
    void increase_method_freq(int method);
    void reset_methods_freq() { methods_freq = std::vector<int>(num_method, 0); }
    
    int num_method;
    int id_selected;
    std::vector<int> methods_freq;
};

class Composer {
public:
    Composer() : target_size(416) {}
    Composer(const char* nanodet_param, const char* nanodet_weight, const char* simplepose_param, const char* simplepose_weight, bool object_stable = true, bool pose_stable = true);
    
    void init(const char* nanodet_param, const char* nanodet_weight, const char* simplepose_param, const char* simplepose_weight, bool object_stable = true, bool pose_stable = true);
    
    void detect(Tensor frame);
    void predict();
    
    void set_object_stabilizer(bool option);
    void set_pose_stabilizer(bool option);
    void set_detection_size(int size);
    
    Tensor get_object_detection() { return stabilized_objects; }
    std::vector<KeyPoint> get_pose_detection() { return keypoints; }
    
    int get_target_index(SelectMethod method, int label);
    void force_set_target_id(int id);
    
private:
    int target_size;
    
    bool enable_object_stabilizer;
    bool enable_pose_stabilizer;
    
    Tensor stabilized_objects;
    std::vector<KeyPoint> keypoints;
    
    Net nanodet;
    Net simplepose;
    
    otter::core::Stabilizer object_stabilizer;
    otter::cv::PoseStabilizer pose_stabilizer;
    
    otter::core::Observer observer;
    Selector selector;
    Suggester suggester;
};

std::vector<Object> from_tensor_to_object(Tensor& objs);

Tensor from_object_to_tensor(std::vector<Object> objs);

Tensor from_trackingbox_to_tensor(std::vector<core::TrackingBox> objs);


}   // end namespace cv
}   // end namespace otter

#endif /* Composer_hpp */
