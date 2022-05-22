//
//  Stabilizer.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/5/13.
//

#ifndef Stabilizer_hpp
#define Stabilizer_hpp

#include <set>
#include <vector>

#include "KalmanTracker.hpp"
#include "GraphicAPI.hpp"
#include "DrawDetection.hpp"
#include "Hungarian.hpp"
#include "Stabilizer.hpp"

namespace otter {
namespace core {

static double GetIOU(otter::cv::Rect_<float> bb_test, otter::cv::Rect_<float> bb_gt) {
    float in = (bb_test & bb_gt).area();
    float un = bb_test.area() + bb_gt.area() - in;

    if (un < DBL_EPSILON)
        return 0;

    return (double)(in / un);
}

typedef struct TrackingBox {
    int frame;
    int id;
    otter::Object obj;
//    otter::cv::Rect_<float> box;
} TrackingBox;

class Stabilizer {
public:
    Stabilizer() {}
    
    std::vector<TrackingBox> track(std::vector<otter::Object> detected_objs) {
        total_frames++;
        frame_count++;
        
        if (detected_objs.size() == 0)
            return std::vector<TrackingBox>();
        
        if (trackers.size() == 0) {
            for (int i = 0 ; i < detected_objs.size(); ++i) {
                otter::cv::KalmanTracker trk = otter::cv::KalmanTracker(detected_objs[i]);
                trackers.push_back(trk);
            }
        }
        
        predictedBoxes.clear();

        for (auto it = trackers.begin(); it != trackers.end();) {
            otter::cv::Rect_<float> pBox = (*it).predict();
            if (pBox.x >= 0 && pBox.y >= 0) {
                predictedBoxes.push_back(pBox);
                it++;
            } else {
                it = trackers.erase(it);
            }
        }
        
        trkNum = predictedBoxes.size();
        detNum = detected_objs.size();

        iouMatrix.clear();
        iouMatrix.resize(trkNum, std::vector<double>(detNum, 0));

        for (unsigned int i = 0; i < trkNum; i++) // compute iou matrix as a distance matrix
        {
            for (unsigned int j = 0; j < detNum; j++) {
                // use 1-iou because the hungarian algorithm computes a minimum-cost assignment.
                iouMatrix[i][j] = 1 - GetIOU(predictedBoxes[i], detected_objs[j].rect);
            }
        }

        // solve the assignment problem using hungarian algorithm.
        // the resulting assignment is [track(prediction) : detection], with len=preNum
        HungarianAlgorithm HungAlgo;
        assignment.clear();
        HungAlgo.Solve(iouMatrix, assignment);

        // find matches, unmatched_detections and unmatched_predictions
        unmatchedTrajectories.clear();
        unmatchedDetections.clear();
        allItems.clear();
        matchedItems.clear();

        if (detNum > trkNum) //    there are unmatched detections
        {
            for (unsigned int n = 0; n < detNum; n++)
                allItems.insert(n);

            for (unsigned int i = 0; i < trkNum; ++i)
                matchedItems.insert(assignment[i]);

            set_difference(allItems.begin(), allItems.end(),
                matchedItems.begin(), matchedItems.end(),
                insert_iterator<set<int>>(unmatchedDetections, unmatchedDetections.begin()));
        } else if (detNum < trkNum) // there are unmatched trajectory/predictions
        {
            for (unsigned int i = 0; i < trkNum; ++i)
                if (assignment[i] == -1) // unassigned label will be set as -1 in the assignment algorithm
                    unmatchedTrajectories.insert(i);
        }

        matchedPairs.clear();
        for (unsigned int i = 0; i < trkNum; ++i)
        {
            if (assignment[i] == -1) // pass over invalid values
                continue;
            if (1 - iouMatrix[i][assignment[i]] < iouThreshold) {
                unmatchedTrajectories.insert(i);
                unmatchedDetections.insert(assignment[i]);
            } else {
                matchedPairs.push_back(cv::Point(i, assignment[i]));
            }
        }
        int detIdx, trkIdx;
        for (unsigned int i = 0; i < matchedPairs.size(); i++) {
            trkIdx = matchedPairs[i].x;
            detIdx = matchedPairs[i].y;
            trackers[trkIdx].update(detected_objs[detIdx]);
        }

        // create and initialise new trackers for unmatched detections
        for (auto umd : unmatchedDetections) {
            otter::cv::KalmanTracker tracker = otter::cv::KalmanTracker(detected_objs[umd]);
            trackers.push_back(tracker);
        }

        // get trackers' output
        frameTrackingResult.clear();
        for (auto it = trackers.begin(); it != trackers.end();) {
            if (((*it).m_time_since_update < 1) && ((*it).m_hit_streak >= min_hits || frame_count <= min_hits)) {
                TrackingBox res;
                res.obj = (*it).get_obj();
                res.id = (*it).m_id + 1;
                res.frame = frame_count;
                frameTrackingResult.push_back(res);
                it++;
            } else {
                it++;
            }

            // remove dead tracklet
            if (it != trackers.end() && (*it).m_time_since_update > max_age)
                it = trackers.erase(it);
        }
        
        return frameTrackingResult;
    }
    
private:
    int iter = 0;
    int total_frames = 0;
    int frame_count = 0;
    int max_age = 1;
    int min_hits = 3;
    double iouThreshold = 0.3;
    unsigned int trkNum = 0;
    unsigned int detNum = 0;
    std::vector<otter::cv::KalmanTracker> trackers;
    std::vector<otter::cv::Rect_<float>> predictedBoxes;
    std::vector<std::vector<double>> iouMatrix;
    std::vector<int> assignment;
    std::set<int> unmatchedDetections;
    std::set<int> unmatchedTrajectories;
    std::set<int> allItems;
    std::set<int> matchedItems;
    std::vector<otter::cv::Point> matchedPairs;
    std::vector<TrackingBox> frameTrackingResult;
};

}   // end namespace core
}   // end namespace otter

#endif /* Stabilizer_hpp */
