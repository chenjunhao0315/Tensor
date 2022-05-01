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

template<typename scalar_t> template<typename scalar_t_2> inline
Size_<scalar_t>::operator Size_<scalar_t_2>() const {
    return Size_<scalar_t_2>(static_cast<scalar_t_2>(width), static_cast<scalar_t_2>(height));
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
    
    Rect_(const Rect_& r) = default;
    Rect_(Rect_&& r) noexcept = default;
    
    Rect_& operator = (const Rect_& r) = default;
    Rect_& operator = (Rect_&& r) noexcept = default;
    
    Rect_(const Point_<scalar_t>& org, const Size_<scalar_t>& sz);
    Rect_(const Point_<scalar_t>& pt1, const Point_<scalar_t>& pt2);
    
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
    
    template<typename scalar_t_2> operator Rect_<scalar_t_2>() const;
    
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

template<typename scalar_t>
inline Rect_<scalar_t>::Rect_(const Point_<scalar_t>& org, const Size_<scalar_t>& sz)
    : x(org.x), y(org.y), width(sz.width), height(sz.height) {}

template<typename scalar_t>
inline Rect_<scalar_t>::Rect_(const Point_<scalar_t>& pt1, const Point_<scalar_t>& pt2) {
    x = std::min(pt1.x, pt2.x);
    y = std::min(pt1.y, pt2.y);
    width = std::max(pt1.x, pt2.x) - x;
    height = std::max(pt1.y, pt2.y) - y;
}

template<typename scalar_t> template<typename scalar_t_2>
inline Rect_<scalar_t>::operator Rect_<scalar_t_2>() const
{
    return Rect_<scalar_t_2>(static_cast<scalar_t_2>(x), static_cast<scalar_t_2>(y), static_cast<scalar_t_2>(width), static_cast<scalar_t_2>(height));
}

template<typename scalar_t> static inline
Rect_<scalar_t>& operator &= (Rect_<scalar_t>& a, const Rect_<scalar_t>& b) {
    if (a.empty() || b.empty()) {
        a = Rect();
        return a;
    }
    const Rect_<scalar_t>& Rx_min = (a.x < b.x) ? a : b;
    const Rect_<scalar_t>& Rx_max = (a.x < b.x) ? b : a;
    const Rect_<scalar_t>& Ry_min = (a.y < b.y) ? a : b;
    const Rect_<scalar_t>& Ry_max = (a.y < b.y) ? b : a;
    
    if ((Rx_min.x < 0 && Rx_min.x + Rx_min.width < Rx_max.x) ||
        (Ry_min.y < 0 && Ry_min.y + Ry_min.height < Ry_max.y)) {
        a = Rect();
        return a;
    }
    
    a.width = std::min(Rx_min.width - (Rx_max.x - Rx_min.x), Rx_max.width);
    a.height = std::min(Ry_min.height - (Ry_max.y - Ry_min.y), Ry_max.height);
    a.x = Rx_max.x;
    a.y = Ry_max.y;
    if (a.empty())
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

void colorToRawData(const Color& color, void *buf, otter::ScalarType dtype, int64_t channels, int64_t unroll_to);

template <typename scalar_t, int l>
class Vec_ {
public:
    enum {
        length = l
    };
    
    Vec_() {
        memset(&val, 0, length * sizeof(scalar_t));
    }
    
    Vec_(scalar_t val_1, scalar_t val_2) {
        val[0] = val_1;
        val[1] = val_2;
    }
    
    scalar_t operator[](int index) {
        return val[index];
    }
private:
    scalar_t val[l];
};

typedef Vec_<int, 2> Vec2i;
typedef Vec_<float, 2> Vec2f;

template <typename scalar_t, int l>
inline std::ostream& operator<<(std::ostream& o, Vec_<scalar_t, l>& vec) {
    return o << "(" << vec[0] << ", " << vec[1] << ")";
}

}   // end namespace cv
}   // end namespace otter

#endif /* GraphicAPI_hpp */
