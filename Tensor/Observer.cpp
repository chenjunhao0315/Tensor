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

otter::cv::Vec2f Observer::observe(otter::Tensor& objects) {
    otter::cv::Vec2f vec;
    
    auto target = getTarget(objects);
    
    if (target.defined()) {
        otter::cv::Rect2f obj;
        
        auto target_data = target.accessor<float, 1>();
        
        int label = int(target_data[0]);
        obj.x = target_data[2];
        obj.y = target_data[3];
        obj.width = target_data[4];
        obj.height = target_data[5];
        
        auto method = methods[label];
        
        vec = getVec(method, obj);
    }
    
    return vec;
}

otter::Tensor Observer::getTarget(otter::Tensor &objects) {
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

    return ((target_index >= 0) ? objects[target_index] : otter::Tensor());
}

otter::cv::Vec2f Observer::getVec(ObserveMethod& method, otter::cv::Rect2f& obj) {
    Anchor& anchor = anchors[method.align];
    
    auto obj_ref = getRefPoint(method.ref, obj);
    auto view_align = getAlignPoint(anchor, obj_ref);
    
    std::cout << "view_align: " << view_align << " obj_ref: " << obj_ref << std::endl;
    
    return {obj_ref.x - view_align.x, obj_ref.y - view_align.y};
}

otter::cv::Point2f Observer::getRefPoint(ObservePosition& position, otter::cv::Rect2f& obj) {
    switch (position) {
        case ObservePosition::CENTER:
            return otter::cv::Point2f(obj.x + obj.width / 2, obj.y + obj.height / 2);
        case ObservePosition::LEFT:
            return otter::cv::Point2f(0, obj.y + obj.height / 2);
        case ObservePosition::RIGHT:
            return otter::cv::Point2f(obj.x + obj.width, obj.y + obj.height / 2);
        case ObservePosition::TOP:
            return otter::cv::Point2f(obj.x + obj.height / 2, obj.y);
        case ObservePosition::BOTTOM:
            return otter::cv::Point2f(obj.x + obj.height / 2, obj.y + obj.height);
        default:
            break;
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
        default:
            break;
    }
}


}   // end namespace core
}   // end namespace otter
