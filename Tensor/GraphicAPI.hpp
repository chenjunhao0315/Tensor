//
//  GraphicAPI.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/3/14.
//

#ifndef GraphicAPI_hpp
#define GraphicAPI_hpp

#include <iostream>
#include <cmath>

#include "Dispatch.hpp"

namespace otter {
namespace cv {

template <typename scalar_t>
class Point_ {
public:
    Point_() : x(0), y(0) {}
    Point_(scalar_t x_, scalar_t y_) : x(x_), y(y_) {}
    
    template<typename scalar_t_2>
    operator Point_<scalar_t_2>() const;
    
    scalar_t x;
    scalar_t y;
};

template<typename scalar_t> template<typename scalar_t_2> inline
Point_<scalar_t>::operator Point_<scalar_t_2>() const {
    return Point_<scalar_t_2>(static_cast<scalar_t_2>(x), static_cast<scalar_t_2>(y));
}

template<typename scalar_t> static inline
bool operator == (const Point_<scalar_t>& a, const Point_<scalar_t>& b) {
    return a.x == b.x && a.y == b.y;
}

template<typename scalar_t> static inline
bool operator != (const Point_<scalar_t>& a, const Point_<scalar_t>& b) {
    return a.x != b.x || a.y != b.y;
}

template<typename scalar_t> static inline
Point_<scalar_t>& operator += (Point_<scalar_t>& a, const Point_<scalar_t>& b) {
    a.x += b.x;
    a.y += b.y;
    return a;
}

template<typename scalar_t> static inline
Point_<scalar_t>& operator -= (Point_<scalar_t>& a, const Point_<scalar_t>& b) {
    a.x -= b.x;
    a.y -= b.y;
    return a;
}

template<typename scalar_t> static inline
Point_<scalar_t> operator + (const Point_<scalar_t>& a, const Point_<scalar_t>& b) {
    return Point_<scalar_t>(static_cast<scalar_t>(a.x + b.x), static_cast<scalar_t>(a.y + b.y));
}

template<typename scalar_t> static inline
Point_<scalar_t> operator - (const Point_<scalar_t>& a, const Point_<scalar_t>& b) {
    return Point_<scalar_t>(static_cast<scalar_t>(a.x - b.x), static_cast<scalar_t>(a.y - b.y));
}

template<typename scalar_t> static inline
Point_<scalar_t> operator - (const Point_<scalar_t>& a) {
    return Point_<scalar_t>(static_cast<scalar_t>(-a.x), static_cast<scalar_t>(-a.y));
}

template<typename scalar_t> static inline
double norm(const Point_<scalar_t>& pt) {
    return std::sqrt((double)pt.x * pt.x + (double)pt.y * pt.y);
}

template <typename scalar_t>
inline std::ostream& operator<<(std::ostream& o, const Point_<scalar_t>& p) {
    o << "[" << p.x << ", " << p.y << "]";
    return o;
}

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
    
    inline scalar_t area() const {
        return width * height;
    }
    
    inline double aspectRatio() const {
        return width / static_cast<double>(height);
    }
    
    inline bool empty() const {
        return width <= 0 || height <= 0;
    }
    
    template<typename scalar_t_2>
    operator Size_<scalar_t_2>() const;
    
    scalar_t width;
    scalar_t height;
};

template<typename scalar_t> template<typename scalar_t2> inline
Size_<scalar_t>::operator Size_<scalar_t2>() const {
    return Size_<scalar_t2>(static_cast<scalar_t2>(width), static_cast<scalar_t2>(height));
}

template <typename scalar_t>
inline std::ostream& operator<<(std::ostream& o, const Size_<scalar_t>& s) {
    o << "[" << s.width << " x " << s.height << "]";
    return o;
}

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
    
    inline Point_<scalar_t> top_left() const {
        return Point_<scalar_t>(x, y);
    }
    
    inline Point_<scalar_t> bottom_right() const {
        return Point_<scalar_t>(x + width, y + height);
    }
    
    inline Size_<scalar_t> size() const {
        return Size_<scalar_t>(width, height);
    }
    
    inline scalar_t area() const {
        return width * height;
    }
    
    inline bool empty() const {
        return x <= 0 || y <= 0 || width <= 0 || height <= 0;
    }
    
    inline bool contains(const Point_<scalar_t>& pt) const {
        return x <= pt.x && pt.x < x + width && y <= pt.y && pt.y < y + height;
    }
    
    scalar_t x;
    scalar_t y;
    scalar_t width;
    scalar_t height;
};

typedef Rect_<int> Rect2i;
typedef Rect_<float> Rect2f;
typedef Rect_<double> Rect2d;
typedef Rect2i Rect;

template<typename scalar_t> static inline
Rect_<scalar_t>& operator &= (Rect_<scalar_t>& a, const Rect_<scalar_t>& b) {
    scalar_t x1 = std::max(a.x, b.x);
    scalar_t y1 = std::max(a.y, b.y);
    a.width = std::min(a.x + a.width, b.x + b.width) - x1;
    a.height = std::min(a.y + a.height, b.y + b.height) - y1;
    a.x = x1;
    a.y = y1;
    if( a.width <= 0 || a.height <= 0 )
        a = Rect();
    return a;
}

template<typename scalar_t> static inline
Rect_<scalar_t>& operator |= (Rect_<scalar_t>& a, const Rect_<scalar_t>& b) {
    if (a.empty()) {
        a = b;
    }
    else if (!b.empty()) {
        scalar_t x1 = std::min(a.x, b.x);
        scalar_t y1 = std::min(a.y, b.y);
        a.width = std::max(a.x + a.width, b.x + b.width) - x1;
        a.height = std::max(a.y + a.height, b.y + b.height) - y1;
        a.x = x1;
        a.y = y1;
    }
    return a;
}

template<typename scalar_t> static inline
Rect_<scalar_t> operator & (const Rect_<scalar_t>& a, const Rect_<scalar_t>& b) {
    Rect_<scalar_t> c = a;
    return c &= b;
}

template<typename scalar_t> static inline
Rect_<scalar_t> operator | (const Rect_<scalar_t>& a, const Rect_<scalar_t>& b) {
    Rect_<scalar_t> c = a;
    return c |= b;
}

template<typename scalar_t> static inline
bool operator == (const Rect_<scalar_t>& a, const Rect_<scalar_t>& b) {
    return a.x == b.x && a.y == b.y && a.width == b.width && a.height == b.height;
}

template<typename scalar_t> static inline
bool operator != (const Rect_<scalar_t>& a, const Rect_<scalar_t>& b) {
    return a.x != b.x || a.y != b.y || a.width != b.width || a.height != b.height;
}

template <typename scalar_t>
inline std::ostream& operator<<(std::ostream& o, const Rect_<scalar_t>& rect) {
    return o << "[" << rect.width << " x " << rect.height << " from (" << rect.x << ", " << rect.y << ")]";
}

class RotatedRect {
public:
    RotatedRect();

    RotatedRect(const Point2f& center, const Size2f& size, float angle);

    RotatedRect(const Point2f& point1, const Point2f& point2, const Point2f& point3);
    
    void points(Point2f pts[]) const;
    
    Rect boundingRect() const;
    
    Rect_<float> boundingRect2f() const;
    
    Point2f center;
    
    Size2f size;
    
    float angle;
};

inline RotatedRect::RotatedRect()
    : center(), size(), angle(0) {}

inline RotatedRect::RotatedRect(const Point2f& _center, const Size2f& _size, float _angle)
    : center(_center), size(_size), angle(_angle) {}

class Color {
public:
    Color(double v0, double v1 = 0, double v2 = 0, double v3 = 0);
    
    double val[4];
};

inline Color::Color(double v0, double v1, double v2, double v3) {
    val[0] = v0;
    val[1] = v1;
    val[2] = v2;
    val[3] = v3;
}

void colorToRawData(const Color& color, void *buf, otter::ScalarType dtype, int channels, int unroll_to);



}   // end namespace cv
}   // end namespace otter

#endif /* GraphicAPI_hpp */
