//
//  ROIAlignLayer.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/7/7.
//

#include "ROIAlignLayer.hpp"
#include "Tensor.hpp"
#include "TensorMaker.hpp"
#include "TensorFactory.hpp"
#include "Parallel.hpp"

namespace otter {

ROIAlignLayer::ROIAlignLayer() {}

int ROIAlignLayer::parse_param(LayerOption& option, ParamDict& pd) {
    int version = opt_find_int(option, "version", 1);
    int aligned = opt_find_int(option, "aligned", 0);
    int pooled_width = opt_find_int(option, "pooled_width", 0);
    int pooled_height = opt_find_int(option, "pooled_height", 0);
    float spatial_scale = opt_find_float(option, "spatial_scale", 1.f);
    float sampling_ratio = opt_find_float(option, "sampling_ratio", 0.f);
    
    pd.set((int)ROIAlignParam::Version, version);
    pd.set((int)ROIAlignParam::Aligned, aligned);
    pd.set((int)ROIAlignParam::PooledWidth, pooled_width);
    pd.set((int)ROIAlignParam::PooledHeight, pooled_height);
    pd.set((int)ROIAlignParam::SpatialScale, spatial_scale);
    pd.set((int)ROIAlignParam::SamplingRatio, sampling_ratio);
    
    return 0;
}

int ROIAlignLayer::compute_output_shape(ParamDict& pd) {
    auto shape_a = bottom_shapes[0].accessor<int, 2>()[0];
    int input_channels = shape_a[1];
    auto shape_b = bottom_shapes[1].accessor<int, 2>()[0];
    int output_batch = shape_b[2];
    
    int pooled_width = pd.get((int)ROIAlignParam::PooledWidth, 0);
    int pooled_height = pd.get((int)ROIAlignParam::PooledHeight, 0);
    
    pd.set(OUTPUT_SHAPE_HINT, otter::tensor({output_batch, input_channels, pooled_height, pooled_width}, ScalarType::Int).view({1, -1}));
    
    return 0;
}

int ROIAlignLayer::load_param(const ParamDict &pd) {
    version = pd.get((int)ROIAlignParam::Version, 1);
    aligned = pd.get((int)ROIAlignParam::Aligned, 0);
    pooled_width = pd.get((int)ROIAlignParam::PooledWidth, 0);
    pooled_height = pd.get((int)ROIAlignParam::PooledHeight, 0);
    spatial_scale = pd.get((int)ROIAlignParam::SpatialScale, 1.f);
    sampling_ratio = pd.get((int)ROIAlignParam::SamplingRatio, 0.f);
    
    return 0;
}

static inline float bilinear_interpolate(const float* ptr, int w, int h, float x, float y) {
    int x0 = (int)x;
    int x1 = x0 + 1;
    int y0 = (int)y;
    int y1 = y0 + 1;

    float a0 = x1 - x;
    float a1 = x - x0;
    float b0 = y1 - y;
    float b1 = y - y0;

    if (x1 >= w)
    {
        x1 = w - 1;
        a0 = 1.f;
        a1 = 0.f;
    }
    if (y1 >= h)
    {
        y1 = h - 1;
        b0 = 1.f;
        b1 = 0.f;
    }

    float r0 = ptr[y0 * w + x0] * a0 + ptr[y0 * w + x1] * a1;
    float r1 = ptr[y1 * w + x0] * a0 + ptr[y1 * w + x1] * a1;

    float v = r0 * b0 + r1 * b1;

    return v;
}

template<typename T>
struct PreCalc {
    int pos1;
    int pos2;
    int pos3;
    int pos4;
    T w1;
    T w2;
    T w3;
    T w4;
};

template<typename T>
void detectron2_pre_calc_for_bilinear_interpolate(
    const int height,
    const int width,
    const int pooled_height,
    const int pooled_width,
    const int iy_upper,
    const int ix_upper,
    T roi_start_h,
    T roi_start_w,
    T bin_size_h,
    T bin_size_w,
    int roi_bin_grid_h,
    int roi_bin_grid_w,
    std::vector<PreCalc<T> >& pre_calc) {
    int pre_calc_index = 0;
    for (int ph = 0; ph < pooled_height; ph++)
    {
        for (int pw = 0; pw < pooled_width; pw++)
        {
            for (int iy = 0; iy < iy_upper; iy++)
            {
                const T yy = roi_start_h + ph * bin_size_h + static_cast<T>(iy + .5f) * bin_size_h / static_cast<T>(roi_bin_grid_h); // e.g., 0.5, 1.5
                for (int ix = 0; ix < ix_upper; ix++)
                {
                    const T xx = roi_start_w + pw * bin_size_w + static_cast<T>(ix + .5f) * bin_size_w / static_cast<T>(roi_bin_grid_w);

                    T x = xx;
                    T y = yy;
                    // deal with: inverse elements are out of feature map boundary
                    if (y < -1.0 || y > height || x < -1.0 || x > width)
                    {
                        // empty
                        PreCalc<T> pc;
                        pc.pos1 = 0;
                        pc.pos2 = 0;
                        pc.pos3 = 0;
                        pc.pos4 = 0;
                        pc.w1 = 0;
                        pc.w2 = 0;
                        pc.w3 = 0;
                        pc.w4 = 0;
                        pre_calc[pre_calc_index++] = pc;
                        continue;
                    }

                    if (y <= 0)
                    {
                        y = 0;
                    }
                    if (x <= 0)
                    {
                        x = 0;
                    }

                    int y_low = (int)y;
                    int x_low = (int)x;
                    int y_high;
                    int x_high;

                    if (y_low >= height - 1)
                    {
                        y_high = y_low = height - 1;
                        y = (T)y_low;
                    }
                    else
                    {
                        y_high = y_low + 1;
                    }

                    if (x_low >= width - 1)
                    {
                        x_high = x_low = width - 1;
                        x = (T)x_low;
                    }
                    else
                    {
                        x_high = x_low + 1;
                    }

                    T ly = y - y_low;
                    T lx = x - x_low;
                    T hy = (T)(1. - ly), hx = (T)(1. - lx);
                    T w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

                    // save weights and indices
                    PreCalc<T> pc;
                    pc.pos1 = y_low * width + x_low;
                    pc.pos2 = y_low * width + x_high;
                    pc.pos3 = y_high * width + x_low;
                    pc.pos4 = y_high * width + x_high;
                    pc.w1 = w1;
                    pc.w2 = w2;
                    pc.w3 = w3;
                    pc.w4 = w4;
                    pre_calc[pre_calc_index++] = pc;
                }
            }
        }
    }
}

template<typename T>
void original_pre_calc_for_bilinear_interpolate(
    const int height,
    const int width,
    const int pooled_height,
    const int pooled_width,
    T roi_start_h,
    T roi_start_w,
    T bin_size_h,
    T bin_size_w,
    int sampling_ratio,
    std::vector<PreCalc<T> >& pre_calc) {
    int pre_calc_index = 0;
    for (int ph = 0; ph < pooled_height; ph++)
    {
        for (int pw = 0; pw < pooled_width; pw++)
        {
            float hstart = roi_start_h + ph * bin_size_h;
            float wstart = roi_start_w + pw * bin_size_w;
            float hend = roi_start_h + (ph + 1) * bin_size_h;
            float wend = roi_start_w + (pw + 1) * bin_size_w;
            hstart = std::min(std::max(hstart, 0.f), (float)height);
            wstart = std::min(std::max(wstart, 0.f), (float)width);
            hend = std::min(std::max(hend, 0.f), (float)height);
            wend = std::min(std::max(wend, 0.f), (float)width);

            int bin_grid_h = (int)(sampling_ratio > 0 ? sampling_ratio : ceil(hend - hstart));
            int bin_grid_w = (int)(sampling_ratio > 0 ? sampling_ratio : ceil(wend - wstart));

            for (int by = 0; by < bin_grid_h; by++)
            {
                float y = hstart + (by + 0.5f) * bin_size_h / (float)bin_grid_h;

                for (int bx = 0; bx < bin_grid_w; bx++)
                {
                    float x = wstart + (bx + 0.5f) * bin_size_w / (float)bin_grid_w;
                    int x0 = (int)x;
                    int x1 = x0 + 1;
                    int y0 = (int)y;
                    int y1 = y0 + 1;

                    float a0 = x1 - x;
                    float a1 = x - x0;
                    float b0 = y1 - y;
                    float b1 = y - y0;

                    if (x1 >= width)
                    {
                        x1 = width - 1;
                        a0 = 1.f;
                        a1 = 0.f;
                    }
                    if (y1 >= height)
                    {
                        y1 = height - 1;
                        b0 = 1.f;
                        b1 = 0.f;
                    }
                    // save weights and indices
                    PreCalc<T> pc;
                    pc.pos1 = y0 * width + x0;
                    pc.pos2 = y0 * width + x1;
                    pc.pos3 = y1 * width + x0;
                    pc.pos4 = y1 * width + x1;
                    pc.w1 = a0 * b0;
                    pc.w2 = a1 * b0;
                    pc.w3 = a0 * b1;
                    pc.w4 = a1 * b1;
                    pre_calc[pre_calc_index++] = pc;
                }
            }
        }
    }
}

int ROIAlignLayer::forward(const std::vector<Tensor>& bottom_blobs, std::vector<Tensor>& top_blobs, const NetOption& /*opt*/) const {
    const Tensor& bottom_blob = bottom_blobs[0];
    int w = bottom_blob.size(3);
    int h = bottom_blob.size(2);
    int channels = bottom_blob.size(1);

    const Tensor& roi_blob = bottom_blobs[1];
    int num_rois = roi_blob.size(0);

    Tensor& top_blob = top_blobs[0];
    top_blob = otter::empty({num_rois, channels, pooled_height, pooled_width}, otter::ScalarType::Float);
    
    auto bottom_blob_a = bottom_blob.accessor<float, 4>()[0];
    auto top_blob_a = top_blob.accessor<float, 4>();
    
    const float* roi_ptr = (const float*)roi_blob.data_ptr();
    for (const auto n_rois : otter::irange(0, num_rois)) {
        
        auto top_blob_roi_a = top_blob_a[n_rois];
        
        float roi_x1 = roi_ptr[1] * spatial_scale;
        float roi_y1 = roi_ptr[2] * spatial_scale;
        float roi_x2 = roi_ptr[3] * spatial_scale;
        float roi_y2 = roi_ptr[4] * spatial_scale;
        roi_ptr += 5;
        if (aligned) {
            roi_x1 -= 0.5f;
            roi_y1 -= 0.5f;
            roi_x2 -= 0.5f;
            roi_y2 -= 0.5f;
        }

        float roi_w = roi_x2 - roi_x1;
        float roi_h = roi_y2 - roi_y1;

        if (!aligned) {
            roi_w = std::max(roi_w, 1.f);
            roi_h = std::max(roi_h, 1.f);
        }

        float bin_size_w = roi_w / (float)pooled_width;
        float bin_size_h = roi_h / (float)pooled_height;
        
        if (version == 0) {
            for (int q = 0; q < channels; q++) {
                const float* ptr = bottom_blob_a[q].data();
                float* outptr = top_blob_roi_a[q].data();

                for (int ph = 0; ph < pooled_height; ph++) {
                    for (int pw = 0; pw < pooled_width; pw++) {
                        // Compute pooling region for this output unit:
                        //  start (included) = ph * roi_height / pooled_height
                        //  end (excluded) = (ph + 1) * roi_height / pooled_height
                        float hstart = roi_y1 + ph * bin_size_h;
                        float wstart = roi_x1 + pw * bin_size_w;
                        float hend = roi_y1 + (ph + 1) * bin_size_h;
                        float wend = roi_x1 + (pw + 1) * bin_size_w;

                        hstart = std::min(std::max(hstart, 0.f), (float)h);
                        wstart = std::min(std::max(wstart, 0.f), (float)w);
                        hend = std::min(std::max(hend, 0.f), (float)h);
                        wend = std::min(std::max(wend, 0.f), (float)w);

                        int bin_grid_h = (int)(sampling_ratio > 0 ? sampling_ratio : ceil(hend - hstart));
                        int bin_grid_w = (int)(sampling_ratio > 0 ? sampling_ratio : ceil(wend - wstart));

                        bool is_empty = (hend <= hstart) || (wend <= wstart);
                        int area = bin_grid_h * bin_grid_w;

                        float sum = 0.f;
                        for (int by = 0; by < bin_grid_h; by++)
                        {
                            float y = hstart + (by + 0.5f) * bin_size_h / (float)bin_grid_h;

                            for (int bx = 0; bx < bin_grid_w; bx++)
                            {
                                float x = wstart + (bx + 0.5f) * bin_size_w / (float)bin_grid_w;

                                // bilinear interpolate at (x,y)
                                float v = bilinear_interpolate(ptr, w, h, x, y);

                                sum += v;
                            }
                        }

                        outptr[pw] = is_empty ? 0.f : (sum / (float)area);
                    }

                    outptr += pooled_width;
                }
            }
        } else if (version == 1) {
            // the version in detectron 2
            int roi_bin_grid_h = (int)(sampling_ratio > 0 ? sampling_ratio : ceil(roi_h / pooled_height));
            int roi_bin_grid_w = (int)(sampling_ratio > 0 ? sampling_ratio : ceil(roi_w / pooled_width));

            const float count = (float)std::max(roi_bin_grid_h * roi_bin_grid_w, 1);

            otter::parallel_for(0, channels, 0, [&](int64_t begin, int64_t end) {
                for (const auto q : otter::irange(begin, end)) {
                    const float* ptr = bottom_blob_a[q].data();
                    float* outptr = top_blob_roi_a[q].data();

                    for (int ph = 0; ph < pooled_height; ph++) {
                        for (int pw = 0; pw < pooled_width; pw++) {
                            float sum = 0.f;
                            for (int by = 0; by < roi_bin_grid_h; by++) {
                                float y = roi_y1 + ph * bin_size_h + (by + 0.5f) * bin_size_h / (float)roi_bin_grid_h;

                                for (int bx = 0; bx < roi_bin_grid_w; bx++) {
                                    float x = roi_x1 + pw * bin_size_w + (bx + 0.5f) * bin_size_w / (float)roi_bin_grid_w;

                                    if (y < -1.0 || y > h || x < -1.0 || x > w) {
                                        // empty
                                        continue;
                                    } else {
                                        if (y <= 0) y = 0;
                                        if (x <= 0) x = 0;

                                        // bilinear interpolate at (x,y)
                                        float v = bilinear_interpolate(ptr, w, h, x, y);
                                        sum += v;
                                    }
                                }
                            }
                            outptr[pw] = sum / count;
                        }

                        outptr += pooled_width;
                    }
                }
            });
        } else if (version == 2) {
            // the version in detectron 2
            // https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/csrc/ROIAlignRotated/ROIAlignRotated_cpu.cpp
            
            int roi_bin_grid_h = (int)(sampling_ratio > 0 ? sampling_ratio : ceil(roi_h / pooled_height));
            int roi_bin_grid_w = (int)(sampling_ratio > 0 ? sampling_ratio : ceil(roi_w / pooled_width));

            const float count = (float)std::max(roi_bin_grid_h * roi_bin_grid_w, 1);

            std::vector<PreCalc<float> > pre_calc(
                (size_t)roi_bin_grid_h * roi_bin_grid_w * pooled_width * pooled_height);
            detectron2_pre_calc_for_bilinear_interpolate(
                h,
                w,
                pooled_height,
                pooled_width,
                roi_bin_grid_h,
                roi_bin_grid_w,
                roi_y1,
                roi_x1,
                bin_size_h,
                bin_size_w,
                roi_bin_grid_h,
                roi_bin_grid_w,
                pre_calc);

            otter::parallel_for(0, channels, 0, [&](int64_t begin, int64_t end) {
                for (const auto q : otter::irange(begin, end)) {
                    const float* ptr = bottom_blob_a[q].data();
                    float* outptr = top_blob_roi_a[q].data();
                    int pre_calc_index = 0;

                    for (int ph = 0; ph < pooled_height; ph++) {
                        for (int pw = 0; pw < pooled_width; pw++) {
                            float output_val = 0.f;
                            for (int iy = 0; iy < roi_bin_grid_h; iy++) {
                                for (int ix = 0; ix < roi_bin_grid_w; ix++) {
                                    PreCalc<float>& pc = pre_calc[pre_calc_index++];

                                    output_val += pc.w1 * ptr[pc.pos1] + pc.w2 * ptr[pc.pos2] + pc.w3 * ptr[pc.pos3] + pc.w4 * ptr[pc.pos4];
                                }
                            }
                            output_val /= count;
                            outptr[pw] = output_val;
                        }
                        outptr += pooled_width;
                    }
                }
            });
        }
    }
    
    return 0;
}

}   // end namespace otter
