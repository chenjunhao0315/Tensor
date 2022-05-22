//
//  KalmanTracker.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/5/13.
//

#include "KalmanTracker.hpp"
#include "TensorMaker.hpp"
#include "TensorOperator.hpp"
#include "DrawDetection.hpp"

namespace otter {
namespace cv {

int KalmanTracker::kf_count = 0;

void KalmanTracker::init_kf(StateType stateMat) {
    int stateNum = 7;
    int measureNum = 4;
    kf = KalmanFilter(stateNum, measureNum, 0, otter::ScalarType::Float);

    measurement = otter::zeros({measureNum, 1}, otter::ScalarType::Float);

    kf.transitionMatrix = otter::tensor({
        1, 0, 0, 0, 1, 0, 0,
        0, 1, 0, 0, 0, 1, 0,
        0, 0, 1, 0, 0, 0, 1,
        0, 0, 0, 1, 0, 0, 0,
        0, 0, 0, 0, 1, 0, 0,
        0, 0, 0, 0, 0, 1, 0,
        0, 0, 0, 0, 0, 0, 1}, otter::ScalarType::Float).view({7, 7});

    kf.measurementMatrix = otter::eye(measureNum, stateNum, otter::ScalarType::Float);
    kf.processNoiseCov = otter::eye(stateNum, stateNum, otter::ScalarType::Float) * 1e-2;
    kf.measurementNoiseCov = otter::eye(measureNum, measureNum, otter::ScalarType::Float) * 1e-1;
    kf.errorCovPost = otter::eye(stateNum, stateNum, otter::ScalarType::Float);

    auto statePost_a = kf.statePost.accessor<float, 2>();
    statePost_a[0][0] = stateMat.x + stateMat.width / 2;
    statePost_a[1][0] = stateMat.y + stateMat.height / 2;
    statePost_a[2][0] = stateMat.area();
    statePost_a[3][0] = stateMat.width / stateMat.height;
}

StateType KalmanTracker::predict() {
    Tensor p = kf.predict();
    auto p_a = p.accessor<float, 2>();
    m_age += 1;

    if (m_time_since_update > 0) {
        m_hit_streak = 0;
    }
    m_time_since_update += 1;

    StateType predictBox = get_rect_xysr(p_a[0][0], p_a[1][0], p_a[2][0], p_a[3][0]);
    
    m_history.push_back(predictBox);
    return m_history.back();
}

void KalmanTracker::update(otter::Object stateObj) {
    m_time_since_update = 0;
    m_history.clear();
    m_hits += 1;
    m_hit_streak += 1;
    
    label = stateObj.label;
    prob = stateObj.prob;
    
    auto measurement_a = measurement.accessor<float, 2>();

    measurement_a[0][0] = stateObj.rect.x + stateObj.rect.width / 2;
    measurement_a[1][0] = stateObj.rect.y + stateObj.rect.height / 2;
    measurement_a[2][0] = stateObj.rect.area();
    measurement_a[3][0] = stateObj.rect.width / stateObj.rect.height;

    kf.correct(measurement);
}

StateType KalmanTracker::get_state() {
    auto s_a = kf.statePost.accessor<float, 2>();
    return get_rect_xysr(s_a[0][0], s_a[1][0], s_a[2][0], s_a[3][0]);
}

StateType KalmanTracker::get_rect_xysr(float cx, float cy, float s, float r) {
    float w = std::sqrt(s * r);
    float h = s / w;
    float x = cx - w / 2;
    float y = cy - h / 2;

    if (x < 0 && cx > 0) {
        x = 0;
    }
    if (y < 0 && cy > 0) {
        y = 0;
    }

    return StateType(x, y, w, h);
}

otter::Object KalmanTracker::get_obj() {
    otter::Object obj;
    obj.rect = get_state();
    obj.label = label;
    obj.prob = prob;
    
    return obj;
}

}   // end namespace cv
}   // end namespace otter
