//
//  Drawing.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/3/13.
//

#ifndef Drawing_hpp
#define Drawing_hpp

#include "Config.hpp"
#include "GraphicAPI.hpp"
#include "DefaultColor.hpp"

namespace otter {
class Tensor;
class Scalar;

namespace cv {

#if OTTER_OPENCV_DRAW

enum LineTypes {
    FILLED  = -1,
    LINE_4  = 4, //!< 4-connected line
    LINE_8  = 8, //!< 8-connected line
    LINE_AA = 16 //!< antialiased line
};

enum HersheyFonts {
    FONT_HERSHEY_SIMPLEX        = 0, //!< normal size sans-serif font
    FONT_HERSHEY_PLAIN          = 1, //!< small size sans-serif font
    FONT_HERSHEY_DUPLEX         = 2, //!< normal size sans-serif font (more complex than FONT_HERSHEY_SIMPLEX)
    FONT_HERSHEY_COMPLEX        = 3, //!< normal size serif font
    FONT_HERSHEY_TRIPLEX        = 4, //!< normal size serif font (more complex than FONT_HERSHEY_COMPLEX)
    FONT_HERSHEY_COMPLEX_SMALL  = 5, //!< smaller version of FONT_HERSHEY_COMPLEX
    FONT_HERSHEY_SCRIPT_SIMPLEX = 6, //!< hand-writing style font
    FONT_HERSHEY_SCRIPT_COMPLEX = 7, //!< more complex variant of FONT_HERSHEY_SCRIPT_SIMPLEX
    FONT_ITALIC                 = 16 //!< flag for italic font
};

void line(Tensor& img, Point pt1, Point pt2, const Color& color, int thickness = 1, int line_type = LINE_8, int shift = 0);

void arrowedLine(Tensor& img, Point pt1, Point pt2, const Color& color, int thickness = 1, int line_type = LINE_8, int shift = 0, double tipLength = 0.1);

void rectangle(Tensor& img, Point pt1, Point pt2, const Color& color, int thickness = 1, int lineType = LINE_8, int shift = 0);

void rectangle(Tensor& img, Rect rec, const Color& color, int thickness = 1, int lineType = LINE_8, int shif = 0);

void circle(Tensor& img, Point center, int radius, const Color& color, int thickness = 1, int line_type = LINE_8, int shift = 0);

void ellipse(Tensor& img, Point center, Size axes, double angle, double start_angle, double end_angle, const Color& color, int thickness = 1, int line_type = LINE_8, int shift = 0);

void ellipse(Tensor& img, const RotatedRect& box, const Color& color, int thickness = 1, int lineType = LINE_8);

void ellipse2Poly(Point2d center, Size2d axes, int angle, int arc_start, int arc_end, int delta, std::vector<Point2d>& pts);

void ellipse2Poly(Point center, Size axes, int angle, int arcStart, int arcEnd, int delta, std::vector<Point>& pts);

void fillConvexPoly(Tensor& img, const Point* pts, int npts, const Color& color, int line_type = LINE_8, int shift = 0);

void fillPoly(Tensor& img, const Point** pts, const int* npts, int ncontours, const Color& color, int line_type = LINE_8, int shift = 0, Point offset = Point());

void polylines(Tensor& img, const Point* const* pts, const int* npts, int ncontours, bool isClosed, const Color& color, int thickness = 1, int line_type = LINE_8, int shift = 0);

void putText(Tensor& img, const std::string& text, Point org, int fontFace, double fontScale, Color color, int thickness = 1, int line_type = LINE_8, bool bottomLeftOrigin = false);

void putText(Tensor& img, const std::string& text, Point org, int FontFace, int pixelHeight, Color color, int thickness = 1, int line_type = LINE_8, bool bottomLeftOrigin = false);

Size getTextSize(const std::string& text, int fontFace, double fontScale, int thickness, int* _base_line);

double getFontScaleFromHeight(const int fontFace, const int pixelHeight, const int thickness);

#endif // OTTER_OPENCV_DRAW

}   // end namespace cv
}   // end namespace otter

#endif /* Drawing_hpp */
