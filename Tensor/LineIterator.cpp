//
//  LineIterator.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/3/14.
//

#include "LineIterator.hpp"

#include "Tensor.hpp"

namespace otter {
namespace cv {

bool clipLine(Size2l img_size, Point2l& pt1, Point2l& pt2) {
    int c1, c2;
    int64_t right  = img_size.width - 1;
    int64_t bottom = img_size.height - 1;
    
    int64_t &x1 = pt1.x, &y1 = pt1.y;
    int64_t &x2 = pt2.x, &y2 = pt2.y;
    
    c1 = (x1 < 0) + (x1 > right) * 2 + (y1 < 0) * 4 + (y1 > bottom) * 8;
    c2 = (x2 < 0) + (x2 > right) * 2 + (y2 < 0) * 4 + (y2 > bottom) * 8;
    
    if( (c1 & c2) == 0 && (c1 | c2) != 0) {
        int64_t a;
        if(c1 & 12) {
            a = c1 < 8 ? 0 : bottom;
            x1 += (int64_t)((double)(a - y1) * (x2 - x1) / (y2 - y1));
            y1 = a;
            c1 = (x1 < 0) + (x1 > right) * 2;
        }
        if( c2 & 12 ) {
            a = c2 < 8 ? 0 : bottom;
            x2 += (int64_t)((double)(a - y2) * (x2 - x1) / (y2 - y1));
            y2 = a;
            c2 = (x2 < 0) + (x2 > right) * 2;
        }
        if( (c1 & c2) == 0 && (c1 | c2) != 0) {
            if(c1) {
                a = c1 == 1 ? 0 : right;
                y1 += (int64_t)((double)(a - x1) * (y2 - y1) / (x2 - x1));
                x1 = a;
                c1 = 0;
            }
            if(c2) {
                a = c2 == 1 ? 0 : right;
                y2 += (int64_t)((double)(a - x2) * (y2 - y1) / (x2 - x1));
                x2 = a;
                c2 = 0;
            }
        }
        
        OTTER_INTERNAL_ASSERT((c1 & c2) != 0 || (x1 | y1 | x2 | y2) >= 0);
    }
    
    return (c1 | c2) == 0;
}

bool clipLine(Size img_size, Point& pt1, Point& pt2) {
    Point2l p1(pt1);
    Point2l p2(pt2);
    bool inside = clipLine(Size2l(img_size.width, img_size.height), p1, p2);
    pt1.x = (int)p1.x;
    pt1.y = (int)p1.y;
    pt2.x = (int)p2.x;
    pt2.y = (int)p2.y;
    
    return inside;
}

bool clipLine(Rect img_rect, Point& pt1, Point& pt2) {
    Point tl = img_rect.top_left();
    pt1 -= tl; pt2 -= tl;
    bool inside = clipLine(img_rect.size(), pt1, pt2);
    pt1 += tl; pt2 += tl;

    return inside;
}

LineIterator::LineIterator(const Tensor& img, Point pt1, Point pt2, int connectivity, bool leftToRight) {
    // TODO: check if HWC
    
    // HWC
    int height = (int)img.size(0);
    int width  = (int)img.size(1);
    init(img, Rect(0, 0, width, height), pt1, pt2, connectivity, leftToRight);
    
    point_mode = false;
}

void LineIterator::init(const Tensor &img, Rect rect, Point pt1_, Point pt2_, int connectivity, bool leftToRight) {
    OTTER_CHECK(connectivity == 8 || connectivity == 4, "Expect the connectivity to be 4 or 8 but get", connectivity);
    
    count = -1;
    p = Point(0, 0);
    ptr = nullptr;
    step = pix_size = 0;
    point_mode = !img.defined();
    
    Point pt1 = pt1_ - rect.top_left();
    Point pt2 = pt2_ - rect.top_left();
    
    if ((pt1.x >= rect.width) || (pt2.x >= rect.width) || (pt1.y >= rect.height) || pt2.y >= rect.height) {
        if (!clipLine(Size(rect.width, rect.height), pt1, pt2)) {
            error = count = 0;
            return;
        }
    }
    
    pt1 += rect.top_left();
    pt2 += rect.top_left();
    
    int delta_x = 1, delta_y = 1;
    int dx = pt2.x - pt1.x;
    int dy = pt2.y - pt1.y;
    
    if (dx < 0) {
        if (leftToRight) {
            dx = -dx;
            dy = -dy;
            pt1 = pt2;
        } else {
            dx = -dx;
            delta_x = -1;
        }
    }
    
    if (dy < 0) {
        dy = -dy;
        delta_y = -1;
    }
    
    bool vert = dy > dx;
    if (vert) {
        std::swap(dx, dy);
        std::swap(delta_x, delta_y);
    }
    
    OTTER_INTERNAL_ASSERT(dx >= 0 && dy >= 0);
    
    if (connectivity == 8) {
        error = dx - (dy + dy);
        plusDelta = dx + dx;
        minusDelta = -(dy + dy);
        minusShift = delta_x;
        plusShift = 0;
        minusStep = 0;
        plusStep = delta_y;
        count = dx + 1;
    }
    else {  // connectivity == 4
        error = 0;
        plusDelta = (dx + dx) + (dy + dy);
        minusDelta = -(dy + dy);
        minusShift = delta_x;
        plusShift = -delta_x;
        minusStep = 0;
        plusStep = delta_y;
        count = dx + dy + 1;
    }
    
    if (vert) {
        std::swap(plusStep, plusShift);
        std::swap(minusStep, minusShift);
    }
    
    p = pt1;
    if (!point_mode) {
        ptr0 = img.data_ptr<unsigned char>();
        pix_size = (int)(img.itemsize() * img.size(2));
        step = (int)(img.size(1) * pix_size);
        ptr = (unsigned char*)ptr0 + (size_t)p.y * step + (size_t)p.x * pix_size;
        plusStep = plusStep * step + plusShift * pix_size;
        minusStep = minusStep * step + minusShift * pix_size;
    }
}

}   // end namespace cv
}   // end namespace otter
