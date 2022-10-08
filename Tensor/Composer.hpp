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
};

std::vector<Object> from_tensor_to_object(Tensor& objs);

Tensor from_object_to_tensor(std::vector<Object> objs);


}   // end namespace cv
}   // end namespace otter

#endif /* Composer_hpp */
