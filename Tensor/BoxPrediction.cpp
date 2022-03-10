#include "BoxPrediction.hpp"

namespace otter {
namespace cv {
    
    Point Center(otter::Tensor& t1, otter::Tensor& t2, int N) {
        float centerX[N]={}, centerY[N]={};
        float weightX = 0, weightY = 0;
        float tot_weight = 0;

        auto weight = t2.accessor<float, 1>();


        auto t1_a = t1.accessor<float, 1>();

        std::vector<BBox> list;

        for (int i = 0; i < N; i++) {
            BBox box;
            box.label = t1_a[6 * i];
            box.score = t1_a[6 * i + 1];
            box.x_high = t1_a[6 * i + 2];
            box.y_high = t1_a[6 * i + 3];
            box.x_low = t1_a[6 * i + 4];
            box.y_low = t1_a[6 * i + 5];
            list.push_back(box);
        }

        for (int i = 0; i < N; i++) {
            centerX[i] = (list[i].x_high + list[i].x_low) / 2;
            centerY[i] = (list[i].y_high + list[i].y_low) / 2;
        }
        for (int i = 0; i < N; i++) {
            weightX = weightX + centerX[i] * weight[list[i].label];
            weightY = weightY + centerY[i] * weight[list[i].label];
            tot_weight = tot_weight + weight[list[i].label];
        }
        weightX = weightX / tot_weight;
        weightY = weightY / tot_weight;


        Point point;
        point.x = weightX;
        point.y = weightY;

        return point;
    }










}
}