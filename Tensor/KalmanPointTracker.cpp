//
//  KalmanPointTracker.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/10/6.
//

#include "KalmanPointTracker.hpp"
#include "TensorMaker.hpp"
#include "TensorOperator.hpp"

namespace otter {
namespace cv {

void KalmanPointTracker::init_kf(Point2f p) {
    int stateNum = 4;
    int measureNum = 2;
    kf = KalmanFilter(stateNum, measureNum, 0, otter::ScalarType::Float);
    
    measurement = otter::zeros({measureNum, 1}, otter::ScalarType::Float);
    
    kf.transitionMatrix = otter::tensor({
        1, 0, 1, 0,
        0, 1, 0, 1,
        0, 0, 1, 0,
        0, 0, 0, 1}, otter::ScalarType::Float).view({4, 4});
    
    kf.measurementMatrix = otter::eye(measureNum, stateNum, otter::ScalarType::Float);
    kf.processNoiseCov = otter::eye(stateNum, stateNum, otter::ScalarType::Float) * 1e-2;
    kf.measurementNoiseCov = otter::eye(measureNum, measureNum, otter::ScalarType::Float) * 5e-1;
    kf.errorCovPost = otter::eye(stateNum, stateNum, otter::ScalarType::Float);
    
    auto statePost_a = kf.statePost.accessor<float, 2>()[0];
    auto statePre_a = kf.statePre.accessor<float, 2>()[0];
    
    statePost_a[0] = p.x;
    statePost_a[1] = p.y;
    statePre_a[0] = p.x;
    statePre_a[1] = p.y;
}

Point2f KalmanPointTracker::predict() {
    Tensor p = kf.predict();
    
    auto point_data = p.accessor<float, 2>()[0];
    
    return Point2f(point_data[0], point_data[1]);
}

void KalmanPointTracker::update(Point2f p) {
    auto measurement_data = measurement.accessor<float, 2>()[0];
    
    measurement_data[0] = p.x;
    measurement_data[1] = p.y;
    
    kf.correct(measurement);
}

}   // end namespace cv
}   // end namepsace otter
