//
//  KalmanFilter.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/4/30.
//

#include "KalmanFilter.hpp"
#include "TensorFactory.hpp"
#include "TensorFunction.hpp"
#include "TensorOperator.hpp"
#include "TensorLinearAlgebra.hpp"

namespace otter {
namespace cv {

KalmanFilter::KalmanFilter() {}

KalmanFilter::KalmanFilter(int dynamParams, int measureParams, int controlParams, otter::ScalarType dtype) {
    init(dynamParams, measureParams, controlParams, dtype);
}

void KalmanFilter::init(int DP, int MP, int CP, otter::ScalarType dtype) {
    OTTER_CHECK(DP > 0 && MP > 0, "Dynamic parameter and measure parameter need greater than 0");
    OTTER_CHECK(dtype == otter::ScalarType::Float || dtype == otter::ScalarType::Double, "Type need to be float or double");
    
    CP = std::max(CP, 0);
    
    statePre = otter::zeros({DP, 1}, dtype);
    statePost = otter::zeros({DP, 1}, dtype);
    transitionMatrix = otter::eye(DP, DP, dtype);
    
    processNoiseCov = otter::eye(DP, DP, dtype);
    measurementMatrix = otter::zeros({MP, DP}, dtype);
    measurementNoiseCov = otter::zeros({MP, MP}, dtype);
    
    errorCovPre = otter::zeros({DP, DP}, dtype);
    errorCovPost = otter::zeros({DP, DP}, dtype);
    gain = otter::zeros({DP, MP}, dtype);

    if (CP > 0)
        controlMatrix = otter::zeros({DP, CP}, dtype);
    else
        controlMatrix.reset();
    
    temp1 = otter::empty({DP, DP}, dtype);
    temp2 = otter::empty({MP, DP}, dtype);
    temp3 = otter::empty({MP, MP}, dtype);
    temp4 = otter::empty({MP, DP}, dtype);
    temp5 = otter::empty({MP, 1}, dtype);
}

const Tensor& KalmanFilter::predict(const Tensor &control) {
    statePre = transitionMatrix.mm(statePost);
    
    if (controlMatrix.defined()) {
        statePre += controlMatrix.mm(control);
    }
    
    otter::native::addmm_out(errorCovPre, processNoiseCov, temp1, transitionMatrix.transpose(0, 1), 1, 1);
    
    statePost.copy_(statePre);
    errorCovPost.copy_(errorCovPre);
    
    return statePre;
}

const Tensor& KalmanFilter::correct(const Tensor &measurement) {
    temp2 = measurementMatrix.mm(errorCovPre);

    otter::native::addmm_out(temp3, measurementNoiseCov, temp2, measurementMatrix.transpose(0, 1), 1, 1);
    
    // temp4 = inv(temp3)*temp2 = Kt(k)
//    temp2 = temp2.contiguous();
//    temp3 = temp3.contiguous();
//    temp4 = temp4.contiguous();
    otter::solve(temp3, temp2, temp4, DECOMP_SVD);
    
    // K(k)
    gain = temp4.transpose(0, 1);
    
    // temp5 = z(k) - H*x'(k)
    temp5 = measurement - measurementMatrix.mm(statePre);
    
    // x(k) = x'(k) + K(k)*temp5
    statePost = statePre + gain.mm(temp5);
    
    // P(k) = P'(k) - K(k)*temp2
    errorCovPost = errorCovPre - gain.mm(temp2);

    return statePost;
}

}   // end namespace cv
}   // end namespace otter
