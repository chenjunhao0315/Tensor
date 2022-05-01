//
//  Observer.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/4/29.
//

#ifndef Observer_hpp
#define Observer_hpp

#include <unordered_map>

#include "GraphicAPI.hpp"
#include "Tensor.hpp"

namespace otter {
namespace core {

enum class AnchorType {
    POINT,
    LINE
};

inline std::ostream& operator<<(std::ostream& o, AnchorType type) {
    switch (type) {
        case AnchorType::POINT:
            return o << "Point";
        case AnchorType::LINE:
            return o << "Line";
        default:
            break;
    }
}

struct Anchor {
    std::string name;
    AnchorType type;
    otter::cv::Point2f pos;
};

inline std::ostream& operator<<(std::ostream& o, Anchor& anchor) {
    return o << "[Anchor] name: " << anchor.name << " type: " << anchor.type << " pos: " << anchor.pos;
}

enum class ObservePosition {
    CENTER,
    LEFT,
    RIGHT,
    TOP,
    BOTTOM
};

struct ObserveMethod {
    int id;
    ObservePosition ref;
    std::string align;
};

class Observer {
public:
    Observer();
    
    void addAnchor(Anchor anchor);
    
    void printAnchor();
    
    void addMethod(ObserveMethod method);
    
    otter::Tensor getTarget(otter::Tensor& objects);
    
    otter::cv::Vec2f observe(otter::Tensor& objects);
    
    otter::cv::Vec2f getVec(ObserveMethod& method, otter::cv::Rect2f& obj);
    
    otter::cv::Point2f getAlignPoint(Anchor& anchor, otter::cv::Point2f& obj);
    
    otter::cv::Point2f getRefPoint(ObservePosition& position, otter::cv::Rect2f& obj);
    
private:
    std::unordered_map<int, ObserveMethod> methods;
    std::unordered_map<std::string, Anchor> anchors;
};

}   // end namespace core
}   // end namespace otter

#endif /* Observer_hpp */
