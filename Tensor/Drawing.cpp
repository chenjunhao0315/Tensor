//
//  Drawing.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/3/13.
//

#include "Drawing.hpp"

#include "Tensor.hpp"
#include "LineIterator.hpp"

namespace otter {
namespace cv {

enum { XY_SHIFT = 16, XY_ONE = 1 << XY_SHIFT, DRAWING_STORAGE_BLOCK = (1 << 12) - 256 };
static const int MAX_THICKNESS = 32767;

static void Line(Tensor& img, Point pt1, Point pt2, const void* _color, int connectivity = 8) {
    if (connectivity == 0)
        connectivity = 8;
    else if (connectivity == 1)
        connectivity = 4;

    LineIterator iterator(img, pt1, pt2, connectivity, true);
    int i, count = iterator.count;
    int pix_size = img.size(2) * img.itemsize();
    const unsigned char* color = (const unsigned char*)_color;

    if (pix_size == 3) {
        for(i = 0; i < count; i++, ++iterator) {
            unsigned char* ptr = *iterator;
            ptr[0] = color[0];
            ptr[1] = color[1];
            ptr[2] = color[2];
        }
    } else {
        for (i = 0; i < count; i++, ++iterator) {
            unsigned char* ptr = *iterator;
            if( pix_size == 1 )
                ptr[0] = color[0];
            else
                memcpy(*iterator, color, pix_size);
        }
    }
}

void line(Tensor& img, Point pt1, Point pt2, const Scalar& color, int thickness, int line_type, int shift) {
    OTTER_CHECK(thickness > 0 && thickness <= MAX_THICKNESS, "Invalid thickness expect 0 < thickness <= ", MAX_THICKNESS);
    OTTER_CHECK(shift >= 0 && shift <= XY_SHIFT, "Invalid shift expect 0 <= shift <= ", XY_SHIFT);

    unsigned char colour[] = {255, 0, 0};
    
    Line(img, pt1, pt2, colour, line_type);
//    double buf[4];
//    scalarToRawData( color, buf, img.type(), 0 );
//    ThickLine( img, pt1, pt2, buf, thickness, line_type, 3, shift );
}

}   // end namespace cv
}   // end namespace otter
