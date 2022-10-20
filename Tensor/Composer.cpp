//
//  Composer.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/10/7.
//

#include "Composer.hpp"

#include "NanodetPlusDetectionOutputLayer.hpp"
#include "TensorTransform.hpp"
#include "PoseEstimation.hpp"

namespace otter {
namespace cv {

int Selector::get_target_index_with_label(Tensor &objects, SelectMethod method, int label) {
    if (method == SelectMethod::MAX_AREA || method == SelectMethod::MAX_AREA_WITH_ID) {
        int target_index = -1;
        float max_area = 0, area;
        
        for (const auto i : otter::irange(objects.size(0))) {
            const auto object = objects[i];
            auto object_a = object.accessor<float, 1>();
            
            if (method == SelectMethod::MAX_AREA_WITH_ID && object_a[6] == id_selected) {
                return i;
            }
            
            if (object_a[0] == label) {
                if ((area = object_a[4] * object_a[5]) > max_area) {
                    max_area = area;
                    target_index = i;
                }
            }
        }
        
        if (method == SelectMethod::MAX_AREA_WITH_ID) {
            id_selected = target_index;
        }
        
        return target_index;
    }
    
    return -1;
}

void Selector::set_id_selected(int id) {
    id_selected = id;
}

static const char *keypoint_name[] = {"nose", "right_eye", "left_eye", "right_ear", "left_ear", "right_shoulder", "left_shoulder", "right_elbow", "left_elbow", "right_wrist", "left_wrist", "right_hip", "left_hip", "right_knee", "left_knee", "right_ankle", "left_ankle"};

void Suggester::increase_method_freq(int method) {
    for (const auto i : otter::irange(num_method)) {
        if (i == method) {
            methods_freq[i] = std::max(methods_freq[i] + 1, 10);
        } else {
            methods_freq[i] = std::min(methods_freq[i] - 2, 0);
        }
    }
}

Movement Suggester::suggest_portrait(Tensor target, std::vector<KeyPoint>& keypoints) {
    Movement move = {otter::cv::Vec2f(0, 0), 90};
    
    float* target_data = target.data_ptr<float>();
    
    int label = target_data[0];
    float x = target_data[2];
    float y = target_data[3];
    float w = target_data[4];
    float h = target_data[5];
    int id = target_data[6];
    
    float area = w * h;
    float center_x = x + w / 2;
    float center_y = y + h / 2;
    
    std::map<std::string, float> keypoints_xy;
    for (const auto i : otter::irange(keypoints.size())) {
        auto keypoint_data = keypoints[i];
        keypoints_xy[std::string(keypoint_name[i])] = keypoint_data.prob > 0.2;
        keypoints_xy[std::string(keypoint_name[i]) + "_x"] = keypoint_data.p.x;
        keypoints_xy[std::string(keypoint_name[i]) + "_y"] = keypoint_data.p.y;
    }

    // Check id
    if (id != id_selected)
        reset_methods_freq();
    
    // Get Method
    int method = 0; // default
    
//    if (keypoints_xy["right_shoulder"] && keypoints_xy["left_shoulder"]) {
//        this->increase_method_freq(1);
//    } else {    // default method
        this->increase_method_freq(0);
//    }
    
    // get highest frequency method
    method = *std::max_element(methods_freq.begin(), methods_freq.end());
    
    // Calculate move
    switch (method) {
        case 0:
            move.vec = otter::cv::Vec2f(center_x - 0.5, center_y - 0.5);
            break;
            
        default:
            break;
    }
    
    return move;
}

Composer::Composer(const char* nanodet_param, const char* nanodet_weight, const char* simplepose_param, const char* simplepose_weight, bool object_stable, bool pose_stable) {
    init(nanodet_param, nanodet_weight, simplepose_param, simplepose_param, object_stable, pose_stable);
}

void Composer::init(const char* nanodet_param, const char* nanodet_weight, const char* simplepose_param, const char* simplepose_weight, bool object_stable, bool pose_stable) {
    nanodet.load_otter(nanodet_param, otter::CompileMode::Inference);
    nanodet.load_weight(nanodet_weight, otter::Net::WeightType::Ncnn);
    
    simplepose.load_otter(simplepose_param, otter::CompileMode::Inference);
    simplepose.load_weight(simplepose_weight, otter::Net::WeightType::Ncnn);
    
    enable_pose_stabilizer = pose_stable;
    enable_object_stabilizer = object_stable;
    target_size = 416;
}

void Composer::set_pose_stabilizer(bool option) {
    enable_pose_stabilizer = option;
}

void Composer::set_object_stabilizer(bool option) {
    enable_object_stabilizer = option;
}

void Composer::set_detection_size(int size) {
    target_size = size;
}

void Composer::detect(Tensor frame) {
    int frame_width = frame.size(3);
    int frame_height = frame.size(2);
    
    float scale;
    int wpad, hpad;
    auto nanodet_pre_process = otter::nanodet_pre_process(frame, target_size, scale, wpad, hpad);
    
    auto nanodet_extractor = nanodet.create_extractor();
        
    nanodet_extractor.input("data_1", nanodet_pre_process);
    otter::Tensor nanodet_predict;
    nanodet_extractor.extract("nanodet", nanodet_predict, 0);
    auto nanodet_post_process = otter::nanodet_post_process(nanodet_predict, frame_width, frame_height, scale, wpad, hpad);
    
    // Normalize width and height
    nanodet_post_process.slice(1, 2, 5, 2) /= frame_width;
    nanodet_post_process.slice(1, 3, 6, 2) /= frame_height;
    
    // Stabilize
    stabilized_objects = nanodet_post_process;
    
    if (enable_object_stabilizer) {
        std::vector<otter::Object> objects = from_tensor_to_object(stabilized_objects);
        
        auto tracking_box = object_stabilizer.track(objects);
        
        stabilized_objects = from_trackingbox_to_tensor(tracking_box);
    }
    
    // Finding the target
    int target_index = this->get_target_index(SelectMethod::MAX_AREA_WITH_ID, 1);
    
    if (target_index != -1) {
        auto object_data = stabilized_objects.accessor<float, 2>();
        
        if (object_data[target_index][0] == 1) {   // If detect the person
            auto target_object = stabilized_objects[target_index].clone();
            target_object.slice(0, 2, 5, 2) *= frame_width;
            target_object.slice(0, 3, 6, 2) *= frame_height;
            
            auto simplepose_input = pose_pre_process(target_object, frame);
                        
            auto simplepose_extractor = simplepose.create_extractor();
            simplepose_extractor.input("data_1", simplepose_input.image);
                        
            otter::Tensor simplepose_predict;
            simplepose_extractor.extract("conv_56", simplepose_predict, 0);
                        
            keypoints = otter::pose_post_process(simplepose_predict, simplepose_input);
            
            // Normalize
            for (auto& keypoint : keypoints) {
                keypoint.p.x /= frame_width;
                keypoint.p.y /= frame_height;
            }
            
            // Stabilize
            if (enable_pose_stabilizer) {
                keypoints = pose_stabilizer.track(keypoints);
            }
        } else {
            keypoints.clear();
        }
    } else {
        pose_stabilizer.reset();
        keypoints.clear();
    }
}

int Composer::get_target_index(SelectMethod method, int label) {
    return selector.get_target_index_with_label(stabilized_objects, method, label);
}

Movement Composer::get_suggest_portrait() {
    int target_index = this->get_target_index(SelectMethod::MAX_AREA_WITH_ID, 1);
    
    if (target_index == -1)
        return {otter::cv::Vec2f(0, 0), 90};
    
    return suggester.suggest_portrait(stabilized_objects[target_index], keypoints);
}

void Composer::force_set_target_id(int id) {
    selector.set_id_selected(id);
}

void Composer::predict() {
    // predict keypoint
    if (enable_pose_stabilizer) {
        keypoints = pose_stabilizer.predict();
    }
}

std::vector<Object> from_tensor_to_object(Tensor& objs) {
    std::vector<Object> objects;
    
    for (int i = 0; i < objs.size(0); ++i) {
        objects.push_back(Object(objs[i]));
    }
    
    return objects;
}

Tensor from_object_to_tensor(std::vector<Object> objs) {
    auto objects = otter::empty({static_cast<long long>(objs.size()), 6}, otter::ScalarType::Float);
    auto object_a = objects.accessor<float, 2>();
    
    for (int i = 0; i < objs.size(); ++i) {
        auto object = object_a[i];
        object[0] = objs[i].label;
        object[1] = objs[i].prob;
        object[2] = objs[i].rect.x;
        object[3] = objs[i].rect.y;
        object[4] = objs[i].rect.width;
        object[5] = objs[i].rect.height;
    }
    
    return objects;
}

Tensor from_trackingbox_to_tensor(std::vector<core::TrackingBox> objs) {
    auto objects = otter::empty({static_cast<long long>(objs.size()), 7}, otter::ScalarType::Float);
    auto object_a = objects.accessor<float, 2>();
    
    for (int i = 0; i < objs.size(); ++i) {
        auto object = object_a[i];
        object[0] = objs[i].obj.label;
        object[1] = objs[i].obj.prob;
        object[2] = objs[i].obj.rect.x;
        object[3] = objs[i].obj.rect.y;
        object[4] = objs[i].obj.rect.width;
        object[5] = objs[i].obj.rect.height;
        object[6] = objs[i].id;
    }
    
    return objects;
}

}   // end namespace cv
}   // end namespace otter
