//
//  LineIterator.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/3/14.
//

#ifndef LineIterator_hpp
#define LineIterator_hpp

#include "GraphicAPI.hpp"

namespace otter {
class Tensor;

namespace cv {

bool clipLine(Size img_size, Point& pt1, Point& pt2);

class LineIterator {
public:
    LineIterator(const Tensor& img, Point pt1, Point pt2, int connectivity = 8, bool leftToRight = false);
    
    void init(const Tensor& img, Rect rect, Point pt1_, Point pt2_, int connectivity, bool leftToRight);
    
    unsigned char* operator*();
    
    LineIterator& operator++();
    
    LineIterator operator++(int);
    
    Point pos() const;
    
    int step, pix_size;
    int count, error;
    int minusDelta, plusDelta;
    int minusStep, plusStep;
    int minusShift, plusShift;
    
    bool point_mode;
    
    Point p;
    
    unsigned char *ptr, *ptr0;
};

inline unsigned char* LineIterator::operator*() {
    return point_mode ? 0 : ptr;
}

inline LineIterator& LineIterator::operator++() {
    int mask = error < 0 ? -1 : 0;
    error += minusDelta + (plusDelta & mask);
    if (!point_mode) {
        ptr += minusStep + (plusStep & mask);
    } else {
        p.x += minusShift + (plusShift & mask);
        p.y += minusStep + (plusStep & mask);
    }
    return *this;
}

inline LineIterator LineIterator::operator++(int) {
    LineIterator it = *this;
    ++(*this);
    return it;
}

inline Point LineIterator::pos() const {
    if (!point_mode) {
        size_t offset = (size_t)(ptr - ptr0);
        int y = (int)(offset / step);
        int x = (int)((offset - (size_t)y * step) / pix_size);
        return Point(x, y);
    }
    return p;
}

}   // end namespace cv
}   // end namespace otter

#endif /* LineIterator_hpp */
