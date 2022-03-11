#ifndef BoxPrediction_hpp
#define BoxPrediction_hpp

#include "Tensor.hpp"
#include "TensorFactory.hpp"

namespace otter {
namespace cv {

using namespace std;

struct BBox {
    int label;
    float score;
    int x_high;
    int x_low;
    int y_high;
    int y_low;
};

struct Point {
    float x;
    float y;
};

Point Center(otter::Tensor& t1, otter::Tensor& t2, int N);











}
}

#endif
