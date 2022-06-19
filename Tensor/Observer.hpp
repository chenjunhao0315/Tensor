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
#include "Interpreter.hpp"

namespace otter {
namespace core {

enum class AnchorType {
    POINT,
    LINE,
    AUTOALIGN
};

inline std::ostream& operator<<(std::ostream& o, AnchorType type) {
    switch (type) {
        case AnchorType::POINT:
            return o << "Point";
        case AnchorType::LINE:
            return o << "Line";
        case AnchorType::AUTOALIGN:
            return o << "Autoalign";
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

enum class ObservePosition : int {
    NOSE                = 0,
    RIGHT_EYE           = 1,
    LEFT_EYE            = 2,
    RIGHT_EAR           = 3,
    LEFT_EAR            = 4,
    RIGHT_SHOULDER      = 5,
    LEFT_SHOULDER       = 6,
    RIGHT_ELBOW         = 7,
    LEFT_ELBOW          = 8,
    RIGHT_WRIST         = 9,
    LEFT_WRIST          = 10,
    RIGHT_HIP           = 11,
    LEFT_HIP            = 12,
    RIGHT_KNEE          = 13,
    LEFT_KNEE           = 14,
    RIGHT_ANKLE         = 15,
    LEFT_ANKLE          = 16,
    CENTER_EYE,
    DOWN_ANKLE,
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

struct Movement {
    otter::cv::Vec2f vec;
    int pitch;
};

class Observer {
public:
    Observer();
    
    void addAnchor(Anchor anchor);
    
    void printAnchor();
    
    void addMethod(ObserveMethod method);
    
    int getTarget(otter::Tensor& objects);
    
    void addCommand(const char* command);
    
    void addTable(std::string name, float value);
    
    void getTable(std::string name);
    
    ObserveMethod getMethod(const otter::Tensor& target, otter::Tensor& objs, otter::Tensor& keypoints);
    
    Movement observe(int index, otter::Tensor& target, otter::Tensor& keypoints);
    
    Movement getVec(ObserveMethod& method, otter::cv::Rect2f& obj, otter::Tensor& keypoints);
    
    otter::cv::Point2f getAlignPoint(Anchor& anchor, otter::cv::Point2f& obj);
    
    otter::cv::Point2f getRefPoint(ObservePosition& position, otter::cv::Rect2f& obj, otter::Tensor& keypoints);
    
private:
    std::unordered_map<int, ObserveMethod> methods;
    std::unordered_map<std::string, Anchor> anchors;
    Interpreter interpreter;
};

}   // end namespace core
}   // end namespace otter

#endif /* Observer_hpp */
