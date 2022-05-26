//
//  Observer.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/4/29.
//

#include "Observer.hpp"
#include "Formatting.hpp"

namespace otter {
namespace core {

Observer::Observer() {
}

void Observer::addAnchor(Anchor anchor) {
    anchors[anchor.name] = anchor;
}

void Observer::printAnchor() {
    for (auto iter : anchors) {
        std::cout << iter.second << std::endl;
    }
}

void Observer::addMethod(ObserveMethod method) {
    methods[method.id] = method;
}

void Observer::addCommand(const char *command) {
    interpreter.addCommand(command, 0);
    interpreter.printCommand();
}

void Observer::addTable(std::string name, float value) {
    interpreter.addTable(name, value);
}

void Observer::getTable(std::string name) {
    interpreter.getTable(name);
}

otter::cv::Vec2f Observer::observe(int target_index, otter::Tensor& objects, otter::Tensor& keypoints) {
    otter::cv::Vec2f vec;
    
    if (objects.defined()) {
        otter::cv::Rect2f obj;
        
        auto target_data = objects.accessor<float, 2>()[target_index];
        
        obj.x = target_data[2];
        obj.y = target_data[3];
        obj.width = target_data[4];
        obj.height = target_data[5];
        
        auto method = getMethod(objects[target_index], objects, keypoints);
        
        vec = getVec(method, obj, keypoints);
    }
    
    return vec;
}

int Observer::getTarget(otter::Tensor &objects) {
    int target_index = -1;
    float max_area = 0, area;
    
    for (const auto i : otter::irange(objects.size(0))) {
        const auto object = objects[i];
        auto object_a = object.accessor<float, 1>();
        
        if (object_a[0] == 1) {
            if ((area = object_a[4] * object_a[5]) > max_area) {
                max_area = area;
                target_index = i;
            }
        }
    }
    
    return target_index;
}

static const char *keypoint_name[] = {"nose", "right_eye", "left_eye", "right_ear", "left_ear", "right_shoulder", "left_shoulder", "right_elbow", "left_elbow", "right_wrist", "left_wrist", "right_hip", "left_hip", "right_knee", "left_knee", "right_ankle", "left_ankle"};

ObserveMethod Observer::getMethod(const otter::Tensor& target, otter::Tensor& objs, otter::Tensor& keypoints) {
    auto target_data = target.accessor<float, 1>();
    
    float w = target_data[4];
    float h = target_data[5];
    float area = w * h;
    
    int method_index = 0;
    
    interpreter.addTable("label", target_data[0]);
    interpreter.addTable("x", target_data[2]);
    interpreter.addTable("y", target_data[3]);
    interpreter.addTable("w", target_data[4]);
    interpreter.addTable("h", target_data[5]);
    interpreter.addTable("area", target_data[4] * target_data[5]);
    interpreter.addTable("center_y", target_data[3] + h / 2);
    interpreter.addTable("method_index", 0);
    interpreter.addTable("success", 0);
    
    if (keypoints.defined()) {
        auto keypoints_data = keypoints.accessor<float, 2>();
        for (int i = 0; i < 17; ++i) {
            auto keypoint_data = keypoints_data[i];
            
            interpreter.addTable(keypoint_name[i], keypoint_data[2] >= 0.2);
            interpreter.addTable(std::string(keypoint_name[i]) + "_x", keypoint_data[0]);
            interpreter.addTable(std::string(keypoint_name[i]) + "_y", keypoint_data[1]);
        }
    }
    
    int success = interpreter.doCommand();
    if (success)
        method_index = interpreter.getTable("method_index");
    
    printf("Area: %g Method index: %d\n", area, method_index);
    
    auto method = methods[method_index];
    
    return method;
}

otter::cv::Vec2f Observer::getVec(ObserveMethod& method, otter::cv::Rect2f& obj, otter::Tensor& keypoints) {
    Anchor& anchor = anchors[method.align];
    
    auto obj_ref = getRefPoint(method.ref, obj, keypoints);
    auto view_align = getAlignPoint(anchor, obj_ref);
    
    if (obj_ref.x == -1 && obj_ref.y == -1) {
        obj_ref.x = view_align.x;
        obj_ref.y = view_align.y;
    }
    
    std::cout << "view_align: " << view_align << " obj_ref: " << obj_ref << std::endl;
    
    return {obj_ref.x - view_align.x, obj_ref.y - view_align.y};
}

otter::cv::Point2f Observer::getRefPoint(ObservePosition& position, otter::cv::Rect2f& obj, otter::Tensor& keypoints) {
    switch (position) {
        case ObservePosition::CENTER:
            return otter::cv::Point2f(obj.x + obj.width / 2, obj.y + obj.height / 2);
        case ObservePosition::LEFT:
            return otter::cv::Point2f(0, obj.y + obj.height / 2);
        case ObservePosition::RIGHT:
            return otter::cv::Point2f(obj.x + obj.width, obj.y + obj.height / 2);
        case ObservePosition::TOP:
            return otter::cv::Point2f(obj.x + obj.width / 2, obj.y);
        case ObservePosition::BOTTOM:
            return otter::cv::Point2f(obj.x + obj.height / 2, obj.y + obj.height);
        case ObservePosition::NOSE:
        case ObservePosition::RIGHT_EYE:
        case ObservePosition::LEFT_EYE:
        case ObservePosition::RIGHT_EAR:
        case ObservePosition::LEFT_EAR:
        case ObservePosition::RIGHT_SHOULDER:
        case ObservePosition::LEFT_SHOULDER:
        case ObservePosition::RIGHT_ELBOW:
        case ObservePosition::LEFT_ELBOW:
        case ObservePosition::RIGHT_WRIST:
        case ObservePosition::LEFT_WRIST:
        case ObservePosition::RIGHT_HIP:
        case ObservePosition::LEFT_HIP:
        case ObservePosition::RIGHT_KNEE:
        case ObservePosition::LEFT_KNEE:
        case ObservePosition::RIGHT_ANKLE:
        case ObservePosition::LEFT_ANKLE: {
            auto keypoint = keypoints.accessor<float, 2>()[int(position)];
            if (keypoint[2] < 0.2)
                return {-1, -1};
            else
                return otter::cv::Point2f(keypoint[0], keypoint[1]);
        }
        case ObservePosition::CENTER_EYE: {
            auto left_eye = keypoints.accessor<float, 2>()[int(ObservePosition::LEFT_EYE)];
            auto right_eye = keypoints.accessor<float, 2>()[int(ObservePosition::RIGHT_EYE)];
            if (left_eye[2] >= 0.2 && right_eye[2] >= 0.2) {
                return {(left_eye[0] + right_eye[0]) / 2, (left_eye[1] + right_eye[1]) / 2};
            } else if (left_eye[2] >= 0.2) {
                return {left_eye[0], left_eye[1]};
            } else if (right_eye[2] >= 0.2) {
                return {right_eye[0], right_eye[1]};
            }
            return {-1, -1};
        }
        case ObservePosition::DOWN_ANKLE: {
            auto left_ankle = keypoints.accessor<float, 2>()[int(ObservePosition::LEFT_ANKLE)];
            auto right_ankle = keypoints.accessor<float, 2>()[int(ObservePosition::RIGHT_ANKLE)];
            if (left_ankle[1] > right_ankle[1]) {
                return {left_ankle[0], left_ankle[1]};
            } else {
                return {right_ankle[0], right_ankle[1]};
            }
        }
    }
    return otter::cv::Point(-1, -1);
}

otter::cv::Point2f Observer::getAlignPoint(Anchor &anchor, otter::cv::Point2f& obj) {
    switch (anchor.type) {
        case AnchorType::POINT:
            return anchor.pos;
        case AnchorType::LINE:
            if (anchor.pos.x == -1)
                return {obj.x, anchor.pos.y};
            return {anchor.pos.x, obj.y};
        case AnchorType::AUTOALIGN:
            if (obj.x > 0.58) {
                return {2. / 3, anchor.pos.y};
            } else if (obj.x > 0.42) {
                return {1. / 2, anchor.pos.y};
            } else {
                return {1. / 3, anchor.pos.y};
            }
        default:
            break;
    }
}


}   // end namespace core
}   // end namespace otter
