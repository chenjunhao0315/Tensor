//
//  GraphicAPI.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/3/14.
//

#ifndef GraphicAPI_hpp
#define GraphicAPI_hpp

#include <iostream>

namespace otter {
namespace cv {

template <typename scalar_t>
class Point_ {
public:
    Point_() : x(0), y(0) {}
    Point_(scalar_t x_, scalar_t y_) : x(x_), y(y_) {}
    
    scalar_t x;
    scalar_t y;
};


typedef Point_<int> Point2i;
typedef Point_<int64_t> Point2l;
typedef Point_<float> Point2f;
typedef Point_<double> Point2d;
typedef Point2i Point;

template <typename scalar_t>
class Size_ {
public:
    Size_() : width(0), height(0) {}
    Size_(scalar_t width_, scalar_t height_) : width(width_), height(height_) {}
    
    scalar_t area() const {
        return width * height;
    }
    
    double aspectRatio() const {
        return width / static_cast<double>(height);
    }
    
    bool empty() const {
        return width <= 0 || height <= 0;
    }
    
    scalar_t width;
    scalar_t height;
};

typedef Size_<int> Size2i;
typedef Size_<int64_t> Size2l;
typedef Size_<float> Size2f;
typedef Size_<double> Size2d;
typedef Size2i Size;

template <typename scalar_t>
class Rect_ {
public:
    Rect_() : x(0), y(0), width(0), height(0) {}
    Rect_(scalar_t x_, scalar_t y_, scalar_t width_, scalar_t height_) : x(x_), y(y_), width(width_), height(height_) {}
    
    Point_<scalar_t> top_left() const {
        return Point_<scalar_t>(x, y);
    }
    
    Point_<scalar_t> bottom_right() const {
        return Point_<scalar_t>(x + width, y + height);
    }
    
    scalar_t area() const {
        return width * height;
    }
    
    bool empty() const {
        return x <= 0 || y <= 0 || width <= 0 || height <= 0;
    }
    
    scalar_t x;
    scalar_t y;
    scalar_t width;
    scalar_t height;
};

template<typename scalar_t> static inline
bool operator == (const Rect_<scalar_t>& a, const Rect_<scalar_t>& b) {
    return a.x == b.x && a.y == b.y && a.width == b.width && a.height == b.height;
}

template<typename scalar_t> static inline
bool operator != (const Rect_<scalar_t>& a, const Rect_<scalar_t>& b) {
    return a.x != b.x || a.y != b.y || a.width != b.width || a.height != b.height;
}

}   // end namespace cv
}   // end namespace otter

#endif /* GraphicAPI_hpp */
