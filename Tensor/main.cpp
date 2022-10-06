#include "OTensor.hpp"
#include "Clock.hpp"
#include "Vision.hpp"
#include "Drawing.hpp"
#include "NanodetPlusDetectionOutputLayer.hpp"
#include "TensorTransform.hpp"
#include "PoseEstimation.hpp"

#include "Composer.hpp"

using namespace std;

int main(int argc, const char * argv[]) {
    
    otter::Clock l;
    auto img = otter::cv::load_image_rgb("5D4A0550cj.jpg");
    l.stop_and_show("ms (read image)");
    
    otter::Clock i;
    auto image_final = img.to(otter::ScalarType::Byte).permute({0, 2, 3, 1}).squeeze(0).contiguous();
    i.stop_and_show("ms (nchw -> nhwc)");
    
    otter::cv::Composer composer("nanodet-plus-m-1.5x_416_fused.otter", "nanodet-plus-m-1.5x_416_fused.bin", "simplepose_fused.otter", "simplepose_fused.bin");
    
    composer.detect(img);
    
    auto objects = composer.get_object_detection();
    objects.slice(1, 2, 5, 2) *= img.size(3);
    objects.slice(1, 3, 6, 2) *= img.size(2);
    
    std::cout << objects << std::endl;
    
    auto keypoints = composer.get_pose_detection();
    
    for (auto& keypoint : keypoints) {
        keypoint.p.x *= img.size(3);
        keypoint.p.y *= img.size(2);
    }
    
    otter::draw_coco_detection(image_final, objects, img.size(3), img.size(2));
    otter::draw_pose_detection(image_final, keypoints);
    
    otter::cv::save_image(image_final, "file");
    
    return 0;
}
