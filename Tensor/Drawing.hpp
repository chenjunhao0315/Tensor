//
//  Drawing.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/3/13.
//

#ifndef Drawing_hpp
#define Drawing_hpp

#include "GraphicAPI.hpp"

namespace otter {
class Tensor;
class Scalar;

namespace cv {

enum LineTypes {
    FILLED  = -1,
    LINE_4  = 4, //!< 4-connected line
    LINE_8  = 8, //!< 8-connected line
    LINE_AA = 16 //!< antialiased line
};

void line(Tensor& img, Point pt1, Point pt2, const Color& color, int thickness, int line_type, int shift);

void arrowedLine(Tensor& img, Point pt1, Point pt2, const Color& color, int thickness, int line_type, int shift, double tipLength);

void rectangle(Tensor& img, Point pt1, Point pt2, const Color& color, int thickness, int lineType, int shift);

void rectangle(Tensor& img, Rect rec, const Color& color, int thickness, int lineType, int shif);

void circle(Tensor& img, Point center, int radius, const Color& color, int thickness, int line_type, int shift);

void ellipse(Tensor& img, Point center, Size axes, double angle, double start_angle, double end_angle, const Color& color, int thickness, int line_type, int shift);

void ellipse(Tensor& img, const RotatedRect& box, const Color& color, int thickness, int lineType);

void fillConvexPoly(Tensor& img, const Point* pts, int npts, const Color& color, int line_type, int shift);

void fillPoly(Tensor& img, const Point** pts, const int* npts, int ncontours, const Color& color, int line_type, int shift, Point offset);

void polylines(Tensor& img, const Point* const* pts, const int* npts, int ncontours, bool isClosed, const Color& color, int thickness, int line_type, int shift);

}   // end namespace cv
}   // end namespace otter

#endif /* Drawing_hpp */
