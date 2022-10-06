//
//  KalmanPointTracker.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/10/6.
//

#ifndef KalmanPointTracker_hpp
#define KalmanPointTracker_hpp

#include "Tensor.hpp"
#include "KalmanFilter.hpp"
#include "TensorFactory.hpp"
#include "GraphicAPI.hpp"

namespace otter {
namespace cv {

class KalmanPointTracker {
public:
    KalmanPointTracker() {
        init_kf(Point2f());
    }
    
    KalmanPointTracker(Point2f p) {
        init_kf(p);
    }
    
    Point2f predict();
    void update(Point2f p);
    
private:
    void init_kf(Point2f p);
    
    KalmanFilter kf;
    Tensor measurement;
};

}   // end namespace cv
}   // end namespace otter

#endif /* KalmanPointTracker_hpp */
