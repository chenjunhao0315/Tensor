#include "OTensor.hpp"
#include "Clock.hpp"
#include "Vision.hpp"
#include "Drawing.hpp"
#include "NanodetPlusDetectionOutputLayer.hpp"
#include "DrawDetection.hpp"
#include "TensorFactory.hpp"
#include "GridSampler.hpp"

using namespace std;

struct Rect {
    float x1;
    float y1;
    float x2;
    float y2;
};

template<typename T>
static void qsort_descent_inplace(std::vector<T>& datas, std::vector<float>& scores);

template<typename T>
static void qsort_descent_inplace(std::vector<T>& datas, std::vector<float>& scores, std::vector<int>& label);

static void nms_sorted_bboxes(const std::vector<Rect>& bboxes, std::vector<size_t>& picked, float nms_threshold);

class AnchorGenerator {
public:
    AnchorGenerator(
        std::vector<std::vector<int>> strides,
        std::vector<int> base_sizes,
        std::vector<float> ratios,
        std::vector<float> scales,
        bool scale_major);
    
    std::vector<otter::Tensor> grid_priors(std::vector<otter::IntArrayRef> featmap_sizes);
    otter::Tensor _get_bboxes_single(std::vector<otter::Tensor>& cls_score_list, std::vector<otter::Tensor>& bbox_pred_list, std::vector<otter::Tensor>& mlvl_priors, otter::IntArrayRef image_shape);
    
private:
    std::vector<otter::Tensor> generate_anchors();
    
    std::vector<otter::Tensor> _meshgrid(otter::Tensor& x, otter::Tensor& y, bool row_major = true);
    otter::Tensor single_level_grid_priors(otter::IntArrayRef featmap_size, int level_idx);
    otter::Tensor _bbox_post_process(std::vector<otter::Tensor>& mlvl_scores, std::vector<otter::Tensor>& mlvl_bboxes, std::vector<otter::Tensor>& mlvl_valid_anchors, std::vector<otter::Tensor>& level_ids, otter::IntArrayRef img_shape);
    
    int num_levels;
    bool scale_major;
    std::vector<std::vector<int>> strides;
    std::vector<int> base_sizes;
    std::vector<float> ratios;
    std::vector<float> scales;
    
    std::vector<otter::Tensor> base_anchors;
};

AnchorGenerator anchor_generator({{4, 4}, {8, 8}, {16, 16}, {32, 32}, {64, 64}}, {4, 8, 16, 32, 64}, {0.5, 1.0, 2.0}, {8.0}, true);

otter::Tensor generate_proposal_lists(std::vector<otter::Tensor>& cls_scores, std::vector<otter::Tensor>& bbox_preds, otter::IntArrayRef image_shape);

otter::Tensor decode(otter::Tensor& bboxes, otter::Tensor& pred_bboxes, otter::IntArrayRef max_shape, std::vector<float> stds_in);

otter::Tensor bbox2roi(otter::Tensor& bboxes);

otter::Tensor map_roi_levels(otter::Tensor& rois, int num_levels);

otter::Tensor extract_index(otter::Tensor& src, otter::Tensor& idx);

void fill_index(otter::Tensor& dst, const otter::Tensor& src, const otter::Tensor& idx);

void get_bboxes(otter::Tensor& det_bboxes, otter::Tensor& det_labels, otter::Tensor& rois, otter::Tensor& cls_score, otter::Tensor& bbox_pred, otter::IntArrayRef image_shape);

void multiclass_nms(otter::Tensor& dets, otter::Tensor& labels, const otter::Tensor& multi_bboxes, const otter::Tensor& multi_scores, float score_thr);

otter::Tensor get_uncertainty(const otter::Tensor& mask_pred, const otter::Tensor& labels);

void get_roi_rel_points_test(otter::Tensor& point_indices, otter::Tensor& rel_roi_points, const otter::Tensor& mask_pred, const otter::Tensor& pred_label);

otter::Tensor _get_fine_grained_point_feats(std::vector<otter::Tensor>& x, otter::Tensor& rois, otter::Tensor& rel_roi_points);

otter::Tensor rel_roi_point_to_rel_img_point(otter::Tensor& rois, otter::Tensor& rel_roi_points, otter::IntArrayRef img, float spatial_scale);

otter::Tensor rel_roi_point_to_abs_img_point(otter::Tensor& rois, otter::Tensor& rel_roi_points);
otter::Tensor abs_img_point_to_rel_img_point(otter::Tensor& abs_img_points, otter::IntArrayRef img, float spatial_scale);

otter::Tensor denormalize(otter::Tensor& grid);

otter::Tensor point_sample(otter::Tensor& input, otter::Tensor& points, bool align_corners = false);

void draw_detection(otter::Tensor& det_bbox, otter::Tensor& det_label, otter::Tensor& img);

void QuickSortFloatD(float* A, int low, int high, int* order);

int main(int argc, const char * argv[]) {
    int ret;
    
    otter::Clock initialize_clock;
    
    otter::Net backbone_neck;
//    backbone_neck.option.use_packing_layout = false;
    backbone_neck.load_otter("backbone+neck-opt.otter", otter::CompileMode::Inference);

    ret = backbone_neck.load_weight("backbone+neck-opt.bin", otter::Net::WeightType::Ncnn);
    if (ret) {
        exit(-1);
    }

    otter::Net rpn;
    rpn.load_otter("rpn-opt.otter", otter::CompileMode::Inference);

    ret = rpn.load_weight("rpn-opt.bin", otter::Net::WeightType::Ncnn);
    if (ret) {
        exit(-1);
    }
    
    otter::Net* bbox_roi_align[4];
    for (int i = 0; i < 4; ++i) {
        float spatial_scale = 0.25 / (1 << i);

        otter::Net* ROIAlign = new otter::Net;
        ROIAlign->addLayer(otter::LayerOption{{"type", "Input"}, {"name", "feature"}, {"output", "feature"}, {"channel", "256"}, {"height", "184"}, {"width", "336"}});
        ROIAlign->addLayer(otter::LayerOption{{"type", "Input"}, {"name", "roi"}, {"input", ""}, {"output", "roi"}, {"channel", "1"}, {"height", "1"}, {"width", "5"}});
        ROIAlign->addLayer(otter::LayerOption{{"type", "ROIAlign"}, {"name", "roi_align"}, {"aligned", "1"}, {"spatial_scale", std::to_string(spatial_scale)}, {"pooled_width", "7"}, {"pooled_height", "7"}, {"version", "1"}, {"input", "feature, roi"}, {"output", "roi_align"}});
        ROIAlign->compile();

        bbox_roi_align[i] = ROIAlign;
    }
    
    otter::Net bbox_head;
    bbox_head.load_otter("bbox_head-opt.otter", otter::CompileMode::Inference);
    
    ret = bbox_head.load_weight("bbox_head-opt.bin", otter::Net::WeightType::Ncnn);
    if (ret) {
        exit(-1);
    }
    
    otter::Net mask_roi_align;
    mask_roi_align.addLayer(otter::LayerOption{{"type", "Input"}, {"name", "feature"}, {"output", "feature"}, {"channel", "256"}, {"height", "184"}, {"width", "336"}});
    mask_roi_align.addLayer(otter::LayerOption{{"type", "Input"}, {"name", "roi"}, {"input", ""}, {"output", "roi"}, {"channel", "1"}, {"height", "1"}, {"width", "5"}});
    mask_roi_align.addLayer(otter::LayerOption{{"type", "ROIAlign"}, {"name", "roi_align"}, {"aligned", "1"}, {"spatial_scale", "0.25"}, {"pooled_width", "14"}, {"pooled_height", "14"}, {"input", "feature, roi"}, {"output", "roi_align"}, {"verions", "0"}});
    mask_roi_align.compile();

    initialize_clock.stop_and_show("ms (initialize)");
    
    otter::Net mask_head;
    mask_head.load_otter("mask_head-opt.otter", otter::CompileMode::Inference);
//    mask_head.summary();
    
    ret = mask_head.load_weight("mask_head-opt.bin", otter::Net::WeightType::Ncnn);
    if (ret) {
        exit(-1);
    }

    FILE *img = fopen("img.bin", "rb");
    fseek(img, 0, SEEK_END);
    size_t size = ftell(img);
    fseek(img, 0, SEEK_SET);
    void* raw_data = malloc(size);
    fread(raw_data, 1, size, img);
    fclose(img);

    otter::Tensor in = otter::from_blob(raw_data, {1, 3, 736, 1344}, otter::ScalarType::Float);

//    otter::Tensor in = otter::ones({1, 3, 736, 1344}, otter::ScalarType::Float);
    
//    auto backbone_profiler = backbone_neck.create_extractor();
//    backbone_profiler.benchmark_info("data_1", "conv_58", {1, 3, 736, 1344});
    
//    auto bbox_head_profiler = bbox_head.create_extractor();
//    bbox_head_profiler.benchmark_info("data_1", "linear_3", {1000, 256, 7, 7});
//    bbox_head_profiler.benchmark_info("data_1", "linear_4", {1000, 256, 7, 7});

    otter::Clock total_clock;
    otter::Clock conv_block;
    
    auto feature_extractor = backbone_neck.create_extractor();
    feature_extractor.input("data_1", in);

    otter::Tensor fpn_0, fpn_1, fpn_2, fpn_3, fpn_4;
    feature_extractor.extract("conv_58", fpn_0, 1);
    feature_extractor.extract("conv_59", fpn_1, 1);
    feature_extractor.extract("conv_60", fpn_2, 1);
    feature_extractor.extract("conv_61", fpn_3, 1);
    feature_extractor.extract("pool_2", fpn_4, 1);

//    cout << fpn_0 << endl;

    std::vector<otter::Tensor> feats;
    feats.push_back(fpn_0);
    feats.push_back(fpn_1);
    feats.push_back(fpn_2);
    feats.push_back(fpn_3);
    feats.push_back(fpn_4);

    otter::Clock rpn_clock;
    std::vector<otter::Tensor> cls_scores, bbox_preds;
    otter::Tensor rpn_0_cls, rpn_0_reg;
    otter::Tensor rpn_1_cls, rpn_1_reg;
    otter::Tensor rpn_2_cls, rpn_2_reg;
    otter::Tensor rpn_3_cls, rpn_3_reg;
    otter::Tensor rpn_4_cls, rpn_4_reg;
    {
        auto rpn_extractor = rpn.create_extractor();
        rpn_extractor.input("data_1", fpn_0);

        rpn_extractor.extract("conv_3", rpn_0_cls, 0);
        rpn_extractor.extract("conv_2", rpn_0_reg, 0);
        cls_scores.push_back(rpn_0_cls);
        bbox_preds.push_back(rpn_0_reg);
    }
    {
        auto rpn_extractor = rpn.create_extractor();
        rpn_extractor.input("data_1", fpn_1);

        rpn_extractor.extract("conv_3", rpn_1_cls, 0);
        rpn_extractor.extract("conv_2", rpn_1_reg, 0);
        cls_scores.push_back(rpn_1_cls);
        bbox_preds.push_back(rpn_1_reg);
    }
    {
        auto rpn_extractor = rpn.create_extractor();
        rpn_extractor.input("data_1", fpn_2);

        rpn_extractor.extract("conv_3", rpn_2_cls, 0);
        rpn_extractor.extract("conv_2", rpn_2_reg, 0);
        cls_scores.push_back(rpn_2_cls);
        bbox_preds.push_back(rpn_2_reg);
    }
    {
        auto rpn_extractor = rpn.create_extractor();
        rpn_extractor.input("data_1", fpn_3);

        rpn_extractor.extract("conv_3", rpn_3_cls, 0);
        rpn_extractor.extract("conv_2", rpn_3_reg, 0);
        cls_scores.push_back(rpn_3_cls);
        bbox_preds.push_back(rpn_3_reg);
    }
    {
        auto rpn_extractor = rpn.create_extractor();
        rpn_extractor.input("data_1", fpn_4);

        rpn_extractor.extract("conv_3", rpn_4_cls, 0);
        rpn_extractor.extract("conv_2", rpn_4_reg, 0);
        cls_scores.push_back(rpn_4_cls);
        bbox_preds.push_back(rpn_4_reg);
    }
    conv_block.stop_and_show("ms (conv block)");

    
    otter::Clock proposal_list_clock;
    otter::Tensor proposal_list = generate_proposal_lists(cls_scores, bbox_preds, {727, 1333});
    proposal_list_clock.stop_and_show("ms (proposal_list)");
    
    otter::Clock bbox_clock;
    // Simple test bbox
    // @param: x
    // @param: poroposal_list
    otter::Tensor det_bbox, det_label;
    {
        // Simple test bbox
        // bbox2roi
        auto rois = bbox2roi(proposal_list);

        if (rois.size(0) == 0) {
            det_bbox = otter::zeros({0, 5}, rois.options());
            det_label = otter::zeros({0, }, otter::ScalarType::Long);
        } else {
            // _bbox_forward
            // bbox_roi_extractor
            // forward
            otter::Tensor bbox_feats;

            int num_levels = 4;
            auto roi_feats = otter::empty({rois.size(0), 256, 7, 7}, rois.options());

            // map_roi_levels
            auto target_lvls = map_roi_levels(rois, num_levels);
            
            for (const auto i : otter::irange(0, num_levels)) {
                auto mask = target_lvls == i;

                int index_count = 0;
                const bool* mask_data = mask.data_ptr<bool>();
                for (const auto j : otter::irange(0, mask.numel())) {
                    if (mask_data[j] == true)
                        index_count++;
                }

                if (index_count > 0) {
                    auto rois_ = extract_index(rois, mask);

                    otter::Net* net = bbox_roi_align[i];
                    auto ex = net->create_extractor();
                    ex.input("feature", feats[i]);
                    ex.input("roi", rois_);

                    otter::Tensor roi_feats_t;
                    ex.extract("roi_align", roi_feats_t, 0);
                    
                    fill_index(roi_feats, roi_feats_t, mask);
                }
            }

            otter::Clock bbox_net_clock;
            auto bbox_extractor = bbox_head.create_extractor();
            bbox_extractor.input("data_1", roi_feats);

            otter::Tensor cls_score, bbox_pred;
            bbox_extractor.extract("linear_3", cls_score, 0);
            bbox_extractor.extract("linear_4", bbox_pred, 0);
            bbox_net_clock.stop_and_show("ms (bbox net)");
            
            get_bboxes(det_bbox, det_label, rois, cls_score, bbox_pred, {727, 1333});
        }
    }
    bbox_clock.stop_and_show("ms (bbox)");
    
    // Simple test mask
    // @param: x, det_bbox, det_label
    otter::Tensor segm_result;
    {
        if (det_bbox.size(0) > 0) {
            auto _bboxes = det_bbox.slice(1, 0, 4, 1);
            
            auto scale_factors = otter::tensor({2.0828, 2.0831, 2.0828, 2.0831}, otter::ScalarType::Float);
            _bboxes = _bboxes * scale_factors;

            auto mask_rois = bbox2roi(_bboxes);
            
            auto mask_roi_extractor = mask_roi_align.create_extractor();
            mask_roi_extractor.input("feature", feats[0]);
            mask_roi_extractor.input("roi", mask_rois);
            
            otter::Tensor mask_feats;   // 1
            mask_roi_extractor.extract("roi_align", mask_feats, 0);
            
            std::vector<otter::Tensor> mask_conv(mask_feats.size(0));
            for (const auto i : otter::irange(0, mask_feats.size(0))) {
                auto mask_extractor = mask_head.create_extractor();
                mask_extractor.input("data_1", mask_feats[i].unsqueeze(0));
                
                mask_extractor.extract("conv_1", mask_conv[i], 0);
            }
            
            auto mask_extractor = mask_head.create_extractor();
            mask_extractor.input("conv_1", otter::native::cat(mask_conv, 0));
            
            otter::Tensor mask_pred;    // 2
            mask_extractor.extract("linear_3", mask_pred, 0);

            mask_pred = mask_pred.view({mask_pred.size(0), 80, 7, 7});
            
            if (det_bbox.size(0) > 0) {
                // _mask_point_forward_test
                
                auto refined_mask_pred = mask_pred.clone();
                
                int subdivision_step = 5;
                int subdivision_num_pionts = 784;
                int scale_factor = 2;
                
                for (int i = 0; i < subdivision_step; ++i) {
                    refined_mask_pred = otter::Interpolate(refined_mask_pred, {refined_mask_pred.size(3) * scale_factor, refined_mask_pred.size(2) * scale_factor}, {double(scale_factor), double(scale_factor)}, otter::InterpolateMode::BILINEAR, false);
                    
                    int num_rois = refined_mask_pred.size(0);
                    int channels = refined_mask_pred.size(1);
                    int mask_height = refined_mask_pred.size(2);
                    int mask_width = refined_mask_pred.size(3);
                    
                    if ((subdivision_num_pionts >= (scale_factor << 1) * mask_height * mask_width) && (i < subdivision_step - 1)) {
                        continue;
                    }
                    
                    otter::Tensor point_indices, rel_roi_points;
                    get_roi_rel_points_test(point_indices, rel_roi_points, refined_mask_pred, det_label);
                    
                    // _get_fine_grained_point_feats
                    auto fine_grained_point_feats = _get_fine_grained_point_feats(feats, mask_rois, rel_roi_points);
                    
                    auto coarse_point_feats = point_sample(mask_pred, rel_roi_points);
                    
                    // head
                    // input: fine_grained_point_feats, coarse_point_feats
                    
                    fine_grained_point_feats.print();
                    coarse_point_feats.print();
                }
            }
        }
    }
    
    
    total_clock.stop_and_show("ms (total)");
    
    auto draw_bbox = det_bbox;
    auto draw_label = det_label;
    auto final = otter::cv::load_image_pixel("/Users/chenjunhao/Desktop/NTHUEE_Project/pointrend/input.jpg");
    draw_detection(draw_bbox, draw_label, final);
    otter::cv::save_image(final, "airplane");

    return 0;
}

otter::Tensor denormalize(otter::Tensor& grid) {
    return grid * 2.0 - 1.0;
}

otter::Tensor _get_fine_grained_point_feats(std::vector<otter::Tensor>& x, otter::Tensor& rois, otter::Tensor& rel_roi_points) {
    auto feats = x[0].packing(1);
    float spatial_scale = 0.25;
    
    auto rel_img_points = rel_roi_point_to_rel_img_point(rois, rel_roi_points, feats.sizes().slice(2), spatial_scale).unsqueeze(0);
    auto point_feat = point_sample(feats, rel_img_points);
    point_feat = point_feat.squeeze(0).transpose(0, 1);
    
    return point_feat;
}

otter::Tensor point_sample(otter::Tensor& input, otter::Tensor& points, bool align_corners) {
    bool add_dim = false;
    
    if (points.dim() == 3) {
        add_dim = true;
        points = points.unsqueeze(2);
    }
    
    auto output = otter::grid_sampler(input, denormalize(points), 0, 0, align_corners);
    
    if (add_dim)
        output = output.squeeze(3);
    
    return output;
}

otter::Tensor rel_roi_point_to_rel_img_point(otter::Tensor& rois, otter::Tensor& rel_roi_points, otter::IntArrayRef img, float spatial_scale) {
    
    auto abs_img_point = rel_roi_point_to_abs_img_point(rois, rel_roi_points);
    auto rel_img_point = abs_img_point_to_rel_img_point(abs_img_point, img, spatial_scale);
    
    return rel_roi_points;
}

otter::Tensor abs_img_point_to_rel_img_point(otter::Tensor& abs_img_points, otter::IntArrayRef img, float spatial_scale) {
    auto scale = otter::tensor({img[1], img[0]}, otter::ScalarType::Float);
    
    return abs_img_points / scale * spatial_scale;
}

otter::Tensor rel_roi_point_to_abs_img_point(otter::Tensor& rois, otter::Tensor& rel_roi_points) {
    if (rois.size(1) == 5) {
        rois = rois.slice(0, 1, 5, 1);
    }
    auto abs_img_points = rel_roi_points.clone();
    
    auto abs_img_points_slice = abs_img_points.slice(1, 0, -1, 1);
    auto xs = abs_img_points_slice.slice(1, 0, 1, 1) * (rois.slice(1, 2, 3, 1) - rois.slice(1, 0, 1, 1));
    auto ys = abs_img_points_slice.slice(1, 1, 2, 1) * (rois.slice(1, 3, 4, 1) - rois.slice(1, 1 ,2, 1));
    xs += rois.slice(1, 0, 1, 1);
    ys += rois.slice(1, 1, 2, 1);
    
    abs_img_points = otter::native::stack({xs, ys}, 2);
    
    return abs_img_points;
}

void get_roi_rel_points_test(otter::Tensor& point_indices, otter::Tensor& rel_roi_points, const otter::Tensor& mask_pred, const otter::Tensor& pred_label) {
    int num_points = 784;
    
    auto uncertainty_map = get_uncertainty(mask_pred, pred_label);
    
    int num_rois = uncertainty_map.size(0);
    int mask_height = uncertainty_map.size(2);
    int mask_width = uncertainty_map.size(3);
    
    float h_step = 1.0 / mask_height;
    float w_step = 1.0 / mask_width;
    
    int mask_size = mask_height * mask_width;
    uncertainty_map = uncertainty_map.view({num_rois, mask_size});
    num_points = std::min(mask_size, num_points);
    
    point_indices = std::get<1>(uncertainty_map.topk(num_points, 1));
    
    auto xs = w_step / 2.0 + (point_indices % mask_width).to(otter::ScalarType::Float) * w_step;
    auto ys = h_step / 2.0 + (point_indices / mask_width).to(otter::ScalarType::Float) * h_step;
    
    rel_roi_points = otter::native::stack({xs, ys}, 2);
}

otter::Tensor get_uncertainty(const otter::Tensor& mask_pred, const otter::Tensor& labels) {
    otter::Tensor gt_class_logits;
    if (mask_pred.size(1) == 1) {
        gt_class_logits = mask_pred.clone();
    } else {
        gt_class_logits = otter::empty({mask_pred.size(0), 1, mask_pred.size(2), mask_pred.size(3)}, otter::ScalarType::Float);
        
        auto labels_a = labels.accessor<int64_t, 2>();
        for (const auto i : otter::irange(0, mask_pred.size(0))) {
            gt_class_logits[i] = mask_pred[i][labels_a[i][0]];
        }
    }
    
    return -otter::native::abs(gt_class_logits);
}

void draw_detection(otter::Tensor& det_bbox, otter::Tensor& det_label, otter::Tensor& img) {
    auto pred = otter::empty({det_bbox.size(0), 6}, otter::ScalarType::Float);
    
    auto pred_a = pred.accessor<float, 2>();
    auto bbox_a = det_bbox.accessor<float, 2>();
    auto label_a = det_label.accessor<int64_t, 2>();
    for (const auto i : otter::irange(0, det_bbox.size(0))) {
        auto pred_s = pred_a[i];
        auto bbox_s = bbox_a[i];
        auto label_s = label_a[i];
        pred_s[0] = label_s[0] + 1;
        pred_s[1] = bbox_s[4];
        pred_s[2] = bbox_s[0];
        pred_s[3] = bbox_s[1];
        pred_s[4] = bbox_s[2] - bbox_s[0];
        pred_s[5] = bbox_s[3] - bbox_s[1];
    }
    
    otter::draw_coco_detection(img, pred, img.size(1), img.size(0));
}

void get_bboxes(otter::Tensor& det_bboxes, otter::Tensor& det_labels, otter::Tensor& rois, otter::Tensor& cls_score, otter::Tensor& bbox_pred, otter::IntArrayRef image_shape) {
    
    auto scores = cls_score.softmax(1);
    auto rois_slice = rois.slice(1, 1, 5, 1);
    
    auto bboxes = decode(rois_slice, bbox_pred, image_shape, {0.1f, 0.1f, 0.2f, 0.2f});
    
    auto scale_factor = otter::tensor({2.0828125, 2.0830946, 2.0828125, 2.0830946}, bboxes.scalar_type());
    bboxes = (bboxes.view({bboxes.size(0), -1, 4}) / scale_factor).view({
        bboxes.sizes()[0], -1});
    
    multiclass_nms(det_bboxes, det_labels, bboxes, scores, 0.05);
}

void multiclass_nms(otter::Tensor& dets, otter::Tensor& labels, const otter::Tensor& multi_bboxes, const otter::Tensor& multi_scores, float score_thr) {
    
    float nms_threshold = 0.5;
    
    int num_classes = multi_scores.size(1) - 1;
    
    otter::Tensor bboxes;
    if (multi_bboxes.size(1) > 4)
        bboxes = multi_bboxes.view({multi_scores.size(0), -1, 4});
        
    auto scores = multi_scores.slice(1, 0, -1, 1);
    
    labels = otter::arange(0, num_classes, 1, otter::ScalarType::Long);
    labels = labels.view({1, -1}).expand_as({scores});

    bboxes = bboxes.reshape({-1, 4});
    scores = scores.reshape({-1});
    labels = labels.reshape({-1});
    
    auto valid_mask = scores > score_thr;
    
    bboxes = extract_index(bboxes, valid_mask);
    scores = extract_index(scores, valid_mask);
    labels = extract_index(labels, valid_mask);
    
    std::vector<Rect> nms_proposal_boxes;
    std::vector<float> nms_scores;
    std::vector<int> nms_label;
    
    float* proposals_data = bboxes.data_ptr<float>();
    float* scores_data = scores.data_ptr<float>();
    int64_t* label_data = labels.data_ptr<int64_t>();
    
    for (const auto i : otter::irange(0, bboxes.size(0))) {
        (void)i;
        nms_proposal_boxes.push_back({proposals_data[0], proposals_data[1], proposals_data[2], proposals_data[3]});
        nms_scores.push_back(scores_data[0]);
        nms_label.push_back(label_data[0]);
        proposals_data += 4;
        scores_data += 1;
        label_data += 1;
    }
    
    qsort_descent_inplace(nms_proposal_boxes, nms_scores, nms_label);
    
    std::vector<size_t> picked;
    nms_sorted_bboxes(nms_proposal_boxes, picked, nms_threshold);
    
    otter::Tensor bboxes_picked = otter::empty({static_cast<long long>(picked.size()), 4}, bboxes.options());
    otter::Tensor scores_picked = otter::empty({static_cast<long long>(picked.size()), 1}, scores.options());
    otter::Tensor labels_picked = otter::empty({static_cast<long long>(picked.size()), 1}, labels.options());
    
    for (const auto i : otter::irange(0, picked.size())) {
        const auto& nms_bboxes = nms_proposal_boxes[picked[i]];
        auto bboxes_data = bboxes_picked[i].accessor<float, 1>();
        auto scores_data = scores_picked[i].accessor<float, 1>();
        auto labels_data = labels_picked[i].accessor<long, 1>();
        
        bboxes_data[0] = nms_bboxes.x1;
        bboxes_data[1] = nms_bboxes.y1;
        bboxes_data[2] = nms_bboxes.x2;
        bboxes_data[3] = nms_bboxes.y2;
        
        scores_data[0] = nms_scores[picked[i]];
        
        labels_data[0] = nms_label[picked[i]];
    }
    
    int max_num = 100;
    
    dets = otter::native::cat({bboxes_picked, scores_picked}, -1).slice(0, 0, max_num, 1);
    labels = labels_picked.slice(0, 0, max_num, 1);
}

void fill_index(otter::Tensor& dst, const otter::Tensor& src, const otter::Tensor& idx) {
    const bool* idx_data = idx.data_ptr<bool>();
    for (int i = 0, j = 0; i < idx.numel(); ++i) {
        if (idx_data[i]) {
            dst[i] = src[j];
            ++j;
        }
    }
}

otter::Tensor extract_index(otter::Tensor& src, otter::Tensor& idx) {
    int index_count = 0;
    const bool* idx_data = idx.data_ptr<bool>();
    
    for (const auto i : otter::irange(0, idx.numel())) {
        if (idx_data[i])
            index_count++;
    }
    
    int dim = src.dim();
    
    otter::Tensor dst;
    if (dim == 1) {
        dst = otter::empty({index_count}, src.options());
    } else if (dim == 2) {
        dst = otter::empty({index_count, src.size(1)}, src.options());
    } else if (dim == 3) {
        dst = otter::empty({index_count, src.size(1), src.size(2)}, src.options());
    } else if (dim == 4) {
        dst = otter::empty({index_count, src.size(1), src.size(2), src.size(3)}, src.options());
    }

    for (int i = 0, j = 0; i < idx.numel(); ++i) {
        if (idx_data[i]) {
            dst[j] = src[i];
            ++j;
        }
    }
    
    return dst;
}

otter::Tensor map_roi_levels(otter::Tensor& rois, int num_levels) {
    otter::Tensor target_lvls = otter::empty({rois.size(0)}, otter::ScalarType::Float);
    
    float* target_lvls_data = (float*)target_lvls.data_ptr();
    const float* rois_data = (const float*)rois.data_ptr();
    
    for (const auto i : otter::irange(0, rois.size(0))) {
        float scale = std::sqrt((rois_data[3] - rois_data[1]) * (rois_data[4] - rois_data[2]));
        float log_floor = std::floor(std::log2(scale / 56 + 1e-6));
        float clamp = std::clamp(log_floor, 0.f, (float)num_levels - 1);
        target_lvls_data[i] = clamp;
        
        rois_data += 5;
    }
    
    return target_lvls.to(otter::ScalarType::Long);
}

otter::Tensor bbox2roi(otter::Tensor& bboxes) {
    otter::Tensor rois;
    if (bboxes.size(0) > 0) {
        auto img_inds = otter::full({bboxes.size(0), 1}, 0, bboxes.options());
        rois = otter::native::cat({img_inds, bboxes.slice(1, 0, 4, 1)}, -1);
    } else {
        rois = otter::zeros({0, 5}, bboxes.options());
    }
    
    return rois;
}

otter::Tensor generate_proposal_lists(std::vector<otter::Tensor>& cls_scores, std::vector<otter::Tensor>& bbox_preds, otter::IntArrayRef image_shape) {
    assert(cls_scores.size() == bbox_preds.size());
    
    int max_per_img = 1000;
    
    std::vector<otter::IntArrayRef> featmap_sizes;
    for (const auto& cls_score : cls_scores) {
        featmap_sizes.push_back(cls_score.sizes().slice(2));
    }
    
    auto mlvl_priors = anchor_generator.grid_priors(featmap_sizes);
    
    otter::Tensor result_list = anchor_generator._get_bboxes_single(cls_scores, bbox_preds, mlvl_priors, image_shape);
    
    return result_list.slice(0, 0, max_per_img, 1);
}

otter::Tensor AnchorGenerator::_get_bboxes_single(std::vector<otter::Tensor>& cls_score_list, std::vector<otter::Tensor>& bbox_pred_list, std::vector<otter::Tensor>& mlvl_anchors, otter::IntArrayRef image_shape) {
    
    int nms_pre = 1000; // TODO: as input
    bool use_sigmoid = true;
    
    std::vector<otter::Tensor> mlvl_scores;
    std::vector<otter::Tensor> mlvl_bbox_preds;
    std::vector<otter::Tensor> mlvl_valid_anchors;
    std::vector<otter::Tensor> level_idxs;
    
    for (int level_idx : otter::irange(0, cls_score_list.size())) {
        auto rpn_cls_score = cls_score_list[level_idx][0];
        auto rpn_bbox_pred = bbox_pred_list[level_idx][0];
        
        rpn_cls_score = rpn_cls_score.permute({1, 2, 0});
        otter::Tensor scores;
        if (use_sigmoid) {
            rpn_cls_score = rpn_cls_score.reshape({-1});
            scores = rpn_cls_score.sigmoid();
        } else {
            assert(false);
        }
        rpn_bbox_pred = rpn_bbox_pred.permute({1, 2, 0}).reshape({-1, 4});
        
        auto anchors = mlvl_anchors[level_idx];
        
        if (0 < nms_pre && nms_pre < scores.size(0)) { // Weird
            float* scores_data = scores.data_ptr<float>();
            auto index = otter::arange(0, scores.size(0), 1, otter::ScalarType::Int);
            int* index_ptr = index.data_ptr<int>();

            QuickSortFloatD(scores_data, 0, scores.size(0) - 1, index_ptr);

            auto new_scores = otter::empty({nms_pre}, otter::ScalarType::Float);
            auto new_rpn_bbox_pred = otter::empty({nms_pre, rpn_bbox_pred.size(1)}, otter::ScalarType::Float);
            auto new_anchors = otter::empty({nms_pre, anchors.size(1)}, otter::ScalarType::Float);
            
            for (int i = 0; i < nms_pre; ++i) {
                new_scores[i] = scores[i];
                new_rpn_bbox_pred[i] = rpn_bbox_pred[index_ptr[i]];
                new_anchors[i] = anchors[index_ptr[i]];
            }
            scores = new_scores;
            rpn_bbox_pred = new_rpn_bbox_pred;
            anchors = new_anchors;
        }

        mlvl_scores.push_back(scores);
        mlvl_bbox_preds.push_back(rpn_bbox_pred);
        mlvl_valid_anchors.push_back(anchors);
        level_idxs.push_back(otter::full({scores.size(0)}, otter::Scalar(level_idx), otter::ScalarType::Long));
    }
    
    return this->_bbox_post_process(mlvl_scores, mlvl_bbox_preds, mlvl_valid_anchors, level_idxs, image_shape);
}

otter::Tensor delta2bbox(otter::Tensor& rois, otter::Tensor& deltas, otter::IntArrayRef max_shape, std::vector<float> stds_in) {
    int num_bboxes = deltas.size(0);
    int num_classes = deltas.size(1) / 4;
    
    if (num_bboxes == 0)
        return deltas;
    
    deltas = deltas.reshape({-1, 4});
    auto means = otter::tensor({0.f, 0.f, 0.f, 0.f}, otter::ScalarType::Float);
    auto stds = otter::tensor(stds_in, otter::ScalarType::Float);
    auto denorm_deltas = deltas * stds + means;
    
    auto dxy = denorm_deltas.slice(1, 0, 2, 1);
    auto dwh = denorm_deltas.slice(1, 2, 4, 1);
    
    auto rois_ = rois.repeat({1, num_classes}).reshape({-1, 4});
    
    otter::Tensor pxy, pwh;
    {
        auto rois_xy = rois_.slice(1, 0, 2, 1);
        auto rois_wh = rois_.slice(1, 2, 4, 1);
        
        pxy = ((rois_xy + rois_wh) * 0.5);
        pwh = (rois_wh - rois_xy);
    }
    
    auto dxy_wh = pwh * dxy;
    
    float max_ratio = 4.135166556742356;    // Maybe constant
    
    dwh = otter::clamp(dwh, -max_ratio, max_ratio);
    
    auto gxy = pxy + dxy_wh;
    auto gwh = pwh * dwh.exp();
    auto x1y1 = gxy - (gwh * 0.5);
    auto x2y2 = gxy + (gwh * 0.5);
    auto bboxes = otter::native::cat({x1y1, x2y2}, -1);
    
    float* bbox = bboxes[0].data_ptr<float>();
    for (const auto i : otter::irange(0, bboxes.size(0))) {
        bbox[0] = std::clamp(bbox[0], (float)0, (float)max_shape[1]);
        bbox[2] = std::clamp(bbox[2], (float)0, (float)max_shape[1]);
        bbox[1] = std::clamp(bbox[1], (float)0, (float)max_shape[0]);
        bbox[3] = std::clamp(bbox[3], (float)0, (float)max_shape[0]);
        bbox += 4;
    }
    
    bboxes = bboxes.reshape({num_bboxes, -1});
    return bboxes;
}

otter::Tensor decode(otter::Tensor& bboxes, otter::Tensor& pred_bboxes, otter::IntArrayRef max_shape, std::vector<float> stds_in) {
    return delta2bbox(bboxes, pred_bboxes, max_shape, stds_in);
}
                            
otter::Tensor AnchorGenerator::_bbox_post_process(std::vector<otter::Tensor>& mlvl_scores, std::vector<otter::Tensor>& mlvl_bboxes, std::vector<otter::Tensor>& mlvl_valid_anchors, std::vector<otter::Tensor>& level_ids, otter::IntArrayRef img_shape) {
    
    float nms_threshold = 0.7;
    
/*
    Use for loop instead of concatance
*/
    
    otter::Tensor det;
    
    for (int q = 0; q < mlvl_scores.size(); ++q) {
        
        auto scores = mlvl_scores[q];
        auto anchors = mlvl_valid_anchors[q];
        auto rpn_bbox_pred = mlvl_bboxes[q];
        
        auto proposals = decode(anchors, rpn_bbox_pred, img_shape, {1.f, 1.f, 1.f, 1.f});
    
        int min_bbox_size = 0;
        
        if (min_bbox_size >= 0) {
            otter::Tensor w = otter::empty({proposals.size(0)}, proposals.options());
            otter::Tensor h = otter::empty({proposals.size(0)}, proposals.options());
            float* w_data = w.data_ptr<float>();
            float* h_data = h.data_ptr<float>();
            for (const auto i : otter::irange(0, proposals.size(0))) {
                float* proposals_data = proposals[i].data_ptr<float>();
                w_data[i] = proposals_data[2] - proposals_data[0];
                h_data[i] = proposals_data[3] - proposals_data[1];
            }
            auto valid_mask = (w > min_bbox_size) & (h > min_bbox_size);
            
            proposals = extract_index(proposals, valid_mask);
            scores = extract_index(scores, valid_mask);
        }
        
        std::vector<Rect> nms_proposal_boxes;
        std::vector<float> nms_scores;
        
        float* proposals_data = proposals.data_ptr<float>();
        float* scores_data = scores.data_ptr<float>();
        
        for (const auto i : otter::irange(0, proposals.size(0))) {
            Rect box;
            box.x1 = proposals_data[0];
            box.y1 = proposals_data[1];
            box.x2 = proposals_data[2];
            box.y2 = proposals_data[3];
            nms_proposal_boxes.push_back(box);
            nms_scores.push_back(scores_data[0]);
            proposals_data += 4;
            scores_data += 1;
        }
        
        qsort_descent_inplace(nms_proposal_boxes, nms_scores);
        
        std::vector<size_t> picked;
        nms_sorted_bboxes(nms_proposal_boxes, picked, nms_threshold);
        
        otter::Tensor proposals_picked = otter::empty({static_cast<long long>(picked.size()), 4}, proposals.options());
        otter::Tensor scores_picked = otter::empty({static_cast<long long>(picked.size()), 1}, scores.options());
        
        for (const auto i : otter::irange(0, picked.size())) {
            auto& nms_proposal_data = nms_proposal_boxes[picked[i]];
            auto proposal_data = proposals_picked[i].accessor<float, 1>();
            auto scores_data = scores_picked[i].accessor<float, 1>();
            
            proposal_data[0] = nms_proposal_data.x1;
            proposal_data[1] = nms_proposal_data.y1;
            proposal_data[2] = nms_proposal_data.x2;
            proposal_data[3] = nms_proposal_data.y2;
            
            scores_data[0] = nms_scores[picked[i]];
        }
        
        if (q == 0) {
            det = otter::native::cat({proposals_picked, scores_picked}, -1);
        } else {
            det = otter::native::cat({det, otter::native::cat({proposals_picked, scores_picked}, -1)}, 0);
        }
    }
    
    // sort det
    std::vector<Rect> nms_proposal_boxes;
    std::vector<float> nms_scores;
    float* det_data = det.data_ptr<float>();
    
    for (const auto i : otter::irange(0, det.size(0))) {
        nms_proposal_boxes.push_back({det_data[0], det_data[1], det_data[2], det_data[3]});
        nms_scores.push_back(det_data[4]);
        det_data += 5;
    }
    qsort_descent_inplace(nms_proposal_boxes, nms_scores);
    
    otter::Tensor sorted_det = otter::empty_like(det);
    float* sorted_det_data = sorted_det.data_ptr<float>();
    for (const auto i : otter::irange(0, det.size(0))) {
        auto& bbox = nms_proposal_boxes[i];
        auto& score = nms_scores[i];
        
        sorted_det_data[0] = bbox.x1;
        sorted_det_data[1] = bbox.y1;
        sorted_det_data[2] = bbox.x2;
        sorted_det_data[3] = bbox.y2;
        sorted_det_data[4] = score;
        sorted_det_data += 5;
    }
           
    return sorted_det.slice(0, 0, 1000, 1);
}

std::vector<otter::Tensor> AnchorGenerator::_meshgrid(otter::Tensor& x, otter::Tensor& y, bool row_major) {
    auto xx = x.repeat({y.size(0)});
    auto yy = y.view({-1, 1}).repeat({1, x.size(0)}).view({-1});
    
    if (row_major)
        return {xx, yy};

    return {yy, xx};
}

std::vector<otter::Tensor> AnchorGenerator::grid_priors(std::vector<otter::IntArrayRef> featmap_sizes) {
    std::vector<otter::Tensor> multi_level_anchors;
    
    for (const auto i : otter::irange(0, this->num_levels)) {
        auto anchors = this->single_level_grid_priors(featmap_sizes[i], i);
        multi_level_anchors.push_back(anchors);
    }
    
    return multi_level_anchors;
}

otter::Tensor AnchorGenerator::single_level_grid_priors(otter::IntArrayRef featmap_size, int level_idx) {
    const auto base_anchors = this->base_anchors[level_idx];
    int feat_h = featmap_size[0];
    int feat_w = featmap_size[1];
    int stride_w = this->strides[level_idx][0];
    int stride_h = this->strides[level_idx][1];
    
    auto shift_x = otter::range(0, feat_w - 1, 1, otter::ScalarType::Float) * stride_w;
    auto shift_y = otter::range(0, feat_h - 1, 1, otter::ScalarType::Float) * stride_h;
    
    auto shift_ = this->_meshgrid(shift_x, shift_y);
    auto shift_xx = shift_[0];
    auto shift_yy = shift_[1];
    
    auto shifts = otter::native::stack({shift_xx, shift_yy, shift_xx, shift_yy}, -1);
    
    otter::Tensor all_anchors = otter::empty({shifts.size(0), base_anchors.size(0), base_anchors.size(1)}, otter::ScalarType::Float);
    for (int i = 0; i < shifts.size(0); ++i) {
        all_anchors[i] = base_anchors + shifts[i];
    }
    
    return all_anchors.view({-1, 4});
}

AnchorGenerator::AnchorGenerator(
    std::vector<std::vector<int>> strides_,
    std::vector<int> base_sizes_,
    std::vector<float> ratios_,
    std::vector<float> scales_,
    bool scale_major_) {
    
    strides = strides_;
    base_sizes = base_sizes_;
    ratios = ratios_;
    scales = scales_;
    scale_major = scale_major_;
    num_levels = base_sizes.size();
    
    base_anchors = this->generate_anchors();
}

std::vector<otter::Tensor> AnchorGenerator::generate_anchors() {
    int num_ratio = ratios.size();
    int num_scale = scales.size();
    
    std::vector<otter::Tensor> base_anchors;
    
    for (int i = 0; i < base_sizes.size(); ++i) {
        otter::Tensor anchors = otter::empty({num_ratio * num_scale, 4}, otter::ScalarType::Float);
        int base_size = base_sizes[i];

        for (int i = 0; i < num_ratio; i++) {
            int w = base_size;
            int h = base_size;
            int cx = 0;   // TODO: as input
            int cy = 0;
            
            float h_ratios = std::sqrt(ratios[i]);
            float w_ratios = 1 / h_ratios;

            for (int j = 0; j < num_scale; j++) {
                float scale = scales[j];
                
                float ws, hs;
                if (scale_major) {
                    ws = w * w_ratios * scale;
                    hs = h * h_ratios * scale;
                } else {
                    ws = w * scale * w_ratios;
                    hs = h * scale * h_ratios;
                }

                float* anchor = (float*)anchors[i * num_scale + j].data_ptr();

                anchor[0] = cx - ws * 0.5f;
                anchor[1] = cy - hs * 0.5f;
                anchor[2] = cx + ws * 0.5f;
                anchor[3] = cy + hs * 0.5f;
            }
        }
        base_anchors.push_back(anchors);
    }
    
    return base_anchors;
}

void QuickSortFloatD(float* A, int low, int high, int* order)
{
    int i = low;
    int j = high;
    float p = A[(low + high) / 2];

    while (i <= j) {
        while (A[i] > p)
            ++i;
        while (A[j] < p)
            --j;
        if (i <= j) {
            std::swap(A[i], A[j]);
            std::swap(order[i], order[j]);
            
            ++i;
            --j;
        }
    }
    if (low < j)
        QuickSortFloatD(A, low, j, order);
    if (i < high)
        QuickSortFloatD(A, i, high, order);
}

static inline float intersection_area(const Rect& a, const Rect& b)
{
    if (a.x1 > b.x2 || a.x2 < b.x1 || a.y1 > b.y2 || a.y2 < b.y1)
    {
        // no intersection
        return 0.f;
    }

    float inter_width = std::min(a.x2, b.x2) - std::max(a.x1, b.x1);
    float inter_height = std::min(a.y2, b.y2) - std::max(a.y1, b.y1);

    return inter_width * inter_height;
}

template<typename T>
static void qsort_descent_inplace(std::vector<T>& datas, std::vector<float>& scores, int left, int right)
{
    int i = left;
    int j = right;
    float p = scores[(left + right) / 2];

    while (i <= j)
    {
        while (scores[i] > p)
            i++;

        while (scores[j] < p)
            j--;

        if (i <= j)
        {
            // swap
            std::swap(datas[i], datas[j]);
            std::swap(scores[i], scores[j]);

            i++;
            j--;
        }
    }

    if (left < j)
        qsort_descent_inplace(datas, scores, left, j);

    if (i < right)
        qsort_descent_inplace(datas, scores, i, right);
}

template<typename T>
static void qsort_descent_inplace(std::vector<T>& datas, std::vector<float>& scores)
{
    if (datas.empty() || scores.empty())
        return;

    qsort_descent_inplace(datas, scores, 0, static_cast<int>(scores.size() - 1));
}

template<typename T>
static void qsort_descent_inplace(std::vector<T>& datas, std::vector<float>& scores, std::vector<int>& label, int left, int right)
{
    int i = left;
    int j = right;
    float p = scores[(left + right) / 2];

    while (i <= j)
    {
        while (scores[i] > p)
            i++;

        while (scores[j] < p)
            j--;

        if (i <= j)
        {
            // swap
            std::swap(datas[i], datas[j]);
            std::swap(scores[i], scores[j]);
            std::swap(label[i], label[j]);

            i++;
            j--;
        }
    }

    if (left < j)
        qsort_descent_inplace(datas, scores, label, left, j);

    if (i < right)
        qsort_descent_inplace(datas, scores, label, i, right);
}

template<typename T>
static void qsort_descent_inplace(std::vector<T>& datas, std::vector<float>& scores, std::vector<int>& label)
{
    if (datas.empty() || scores.empty() || label.empty())
        return;

    qsort_descent_inplace(datas, scores, label, 0, static_cast<int>(scores.size() - 1));
}

static void nms_sorted_bboxes(const std::vector<Rect>& bboxes, std::vector<size_t>& picked, float nms_threshold)
{
    picked.clear();

    const size_t n = bboxes.size();

    std::vector<float> areas(n);
    for (size_t i = 0; i < n; i++)
    {
        const Rect& r = bboxes[i];

        float width = r.x2 - r.x1;
        float height = r.y2 - r.y1;

        areas[i] = width * height;
    }

    for (size_t i = 0; i < n; i++)
    {
        const Rect& a = bboxes[i];

        int keep = 1;
        for (size_t j = 0; j < picked.size(); j++)
        {
            const Rect& b = bboxes[picked[j]];

            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            //             float IoU = inter_area / union_area
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
}
