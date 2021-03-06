//
//  ShortCutLayer.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/3/1.
//

#include "ShortCutLayer.hpp"
#include "TensorOperator.hpp"
#include "TensorFactory.hpp"
#include "TensorEltwise.hpp"

namespace otter {

ShortCutLayer::ShortCutLayer() {
    one_blob_only = false;
    support_inplace = false;
#if __SSE2__
    support_packing = true;
#elif __ARM_NEON__
    support_packing = true;
#endif
}

int ShortCutLayer::parse_param(LayerOption& /*option*/, ParamDict &pd) {
    pd.clear();
    // alpha beta
    
    return 0;
}

int ShortCutLayer::load_param(const ParamDict &/*pd*/) {
    // alpha beta
    
    return 0;
}

int ShortCutLayer::compute_output_shape(ParamDict &pd) {
    pd.set(OUTPUT_SHAPE_HINT, bottom_shapes[0]);
    
    return 0;
}

ShortCutBackend shortcut_check_and_select_backend(const Tensor& self, const Tensor& other) {
    OTTER_CHECK(self.size(0) == self.size(0), "[Shortcut] Batch size should be the same, but get ", self.size(0), " != ", other.size(0));
    OTTER_CHECK(self.size(2) == self.size(2), "[Shortcut] Height should be the same, but get ", self.size(0), " != ", other.size(0));
    OTTER_CHECK(self.size(3) == self.size(3), "[Shortcut] Width size should be the same, but get ", self.size(0), " != ", other.size(0));
    
    return (self.size(1) != other.size(1)) ? ShortCutBackend::Darknet_shortcut : ShortCutBackend::Eltwise_add;
}

int ShortCutLayer::forward(const std::vector<Tensor>& bottom_blobs, std::vector<Tensor>& top_blobs, const NetOption& /*opt*/) const {
    const Tensor& bottom_blob = bottom_blobs[0];
    const Tensor& bottom_blob_next = bottom_blobs[1];
    Tensor& output = top_blobs[0];
    
    int elempack1 = bottom_blob.elempack();
    int elempack2 = bottom_blob_next.elempack();
    
    ShortCutBackend backend = shortcut_check_and_select_backend(bottom_blob, bottom_blob_next);
    
    if (elempack1 == 8 || elempack2 == 8) {
        if (backend == ShortCutBackend::Eltwise_add) {
            if (bottom_blob.dim() == 4 && bottom_blob.size(0) == 1 && bottom_blob_next.dim() == 4 && bottom_blob_next.size(0) == 1) {
                output = eltwise_add_pack8(bottom_blob.squeeze(0), bottom_blob_next.squeeze(0)).unsqueeze_(0);
            } else {
                if (bottom_blob.dim() > 4 || bottom_blob_next.dim() > 4) {
                    output = bottom_blob.packing(1) + bottom_blob_next.packing(1);
                } else {
                    output = eltwise_add_pack8(bottom_blob, bottom_blob_next);
                }
            }
        }
        
        return 0;
    } else if (elempack1 == 4 || elempack2 == 4) {
        if (backend == ShortCutBackend::Eltwise_add) {
            if (bottom_blob.dim() == 4 && bottom_blob.size(0) == 1 && bottom_blob_next.dim() == 4 && bottom_blob_next.size(0) == 1) {
                output = eltwise_add_pack4(bottom_blob.squeeze(0), bottom_blob_next.squeeze(0)).unsqueeze_(0);
            } else {
                if (bottom_blob.dim() > 4 || bottom_blob_next.dim() > 4) {
                    output = bottom_blob.packing(1) + bottom_blob_next.packing(1);
                } else {
                    output = eltwise_add_pack4(bottom_blob, bottom_blob_next);
                }
            }
        }
        
        return 0;
    }
    
    switch (backend) {
        case ShortCutBackend::Darknet_shortcut: break;
        case ShortCutBackend::Eltwise_add: output = bottom_blob + bottom_blob_next; break;
    }
    
    for (size_t i = 2; i < bottom_blobs.size(); ++i) {
        const Tensor& bottom_blob_next = bottom_blobs[i];
        ShortCutBackend backend = shortcut_check_and_select_backend(bottom_blob, bottom_blob_next);
        switch (backend) {
            case ShortCutBackend::Darknet_shortcut: {
                // TODO: darknet version shortcut
                break;
            }
            case ShortCutBackend::Eltwise_add: {
                output = output + bottom_blob_next;
                break;
            }
        }
    }
    
    top_blobs[0] = output;
    
    return 0;
}

}
