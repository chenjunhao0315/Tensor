//
//  PoseStabilizer.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/10/7.
//

#ifndef PoseStabilizer_hpp
#define PoseStabilizer_hpp

#include <vector>

#include "PoseEstimation.hpp"
#include "KalmanPointTracker.hpp"

namespace otter {
namespace cv {

class PoseStabilizer {
public:
    PoseStabilizer() {}
    
    void reset() { kpts.clear(); checklist.clear(); }
    
    std::vector<KeyPoint> track(std::vector<KeyPoint> keypoints);
    
    std::vector<KeyPoint> predict();
    
private:
    std::vector<KalmanPointTracker> kpts;
    std::vector<int> checklist;
};

}   // end namespace cv
}   // end namespace otter

#endif /* PoseStabilizer_hpp */
