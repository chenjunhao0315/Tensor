//
//  KalmanFilter.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/4/30.
//

#ifndef KalmanFilter_hpp
#define KalmanFilter_hpp

#include "Tensor.hpp"

namespace otter {
namespace cv {

class KalmanFilter {
public:
    KalmanFilter();
    KalmanFilter(int dynamParams, int measureParams, int controlParams, otter::ScalarType dtype);
    
    void init(int DP, int MP, int CP, otter::ScalarType dtype);
    
    const Tensor& predict(const Tensor& control = Tensor());
    
    const Tensor& correct(const Tensor& measurement);
    
//private:
    otter::Tensor statePre;
    otter::Tensor statePost;
    otter::Tensor transitionMatrix;
    
    otter::Tensor processNoiseCov;
    otter::Tensor measurementMatrix;
    otter::Tensor measurementNoiseCov;
    otter::Tensor errorCovPre;
    otter::Tensor errorCovPost;
    otter::Tensor gain;
    
    otter::Tensor controlMatrix;
    
    otter::Tensor temp1;
    otter::Tensor temp2;
    otter::Tensor temp3;
    otter::Tensor temp4;
    otter::Tensor temp5;
};

}   // end namespace cv
}   // end namespace otter

#endif /* KalmanFilter_hpp */
