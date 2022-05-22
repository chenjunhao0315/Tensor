//
//  KalmanTracker.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/5/13.
//

#ifndef KalmanTracker_hpp
#define KalmanTracker_hpp

#include "Tensor.hpp"
#include "KalmanFilter.hpp"
#include "TensorFactory.hpp"
#include "AutoBuffer.hpp"
#include "GraphicAPI.hpp"
#include "DrawDetection.hpp"

namespace otter {
namespace cv {

#define StateType Rect_<float>

class KalmanTracker {
public:
    KalmanTracker() {
        init_kf(StateType());
        label = 0;
        prob = 0;
        m_time_since_update = 0;
        m_hits = 0;
        m_hit_streak = 0;
        m_age = 0;
        m_id = kf_count;
    }

    KalmanTracker(otter::Object initObj) {
        init_kf(initObj.rect);
        label = initObj.label;
        prob = initObj.prob;
        m_time_since_update = 0;
        m_hits = 0;
        m_hit_streak = 0;
        m_age = 0;
        m_id = kf_count;
        kf_count++;
    }

    ~KalmanTracker() {
        m_history.clear();
    }
    
    StateType predict();
    void update(otter::Object stateObj);

    StateType get_state();
    StateType get_rect_xysr(float cx, float cy, float s, float r);

    otter::Object get_obj();
    
    static int kf_count;
    int m_time_since_update;
    int m_hits;
    int m_hit_streak;
    int m_age;
    int m_id;

private:
    void init_kf(StateType stateMat);

    KalmanFilter kf;
    Tensor measurement;
    int label;
    float prob;

    std::vector<StateType> m_history;
};

}   // end namespace cv
}   // end namespace otter

#endif /* KalmanTracker_hpp */
