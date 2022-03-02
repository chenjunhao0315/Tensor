//
//  InputLayer.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/2/19.
//

#include "InputLayer.hpp"
#include "LayerRegistry.hpp"

#include "TensorMaker.hpp"

namespace otter {

InputLayer::InputLayer() {
    one_blob_only = true;
    support_inplace = true;
}

int InputLayer::parse_param(LayerOption &option, ParamDict& pd) {
    pd.clear();
    int n = opt_find_int(option, "batchsize", 1);
    int c = opt_find_int(option, "channel", 1);
    int h = opt_find_int(option, "height", 1);
    int w = opt_find_int(option, "width", 1);
    pd.set((int)InputParam::Batch, n);
    pd.set((int)InputParam::Channel, c);
    pd.set((int)InputParam::Height, h);
    pd.set((int)InputParam::Width, w);
    
    return 0;
}

int InputLayer::compute_output_shape(ParamDict &pd) {
    int n = pd.get((int)InputParam::Batch, 0);
    int c = pd.get((int)InputParam::Channel, 0);
    int h = pd.get((int)InputParam::Height, 0);
    int w = pd.get((int)InputParam::Width, 0);
    pd.set(OUTPUT_SHAPE_HINT, tensor({n, c, h, w}, ScalarType::Int));
    
    return 0;
}

int InputLayer::load_param(const ParamDict& pd) {
    int n = pd.get((int)InputParam::Batch, 0);
    int c = pd.get((int)InputParam::Channel, 0);
    int h = pd.get((int)InputParam::Height, 0);
    int w = pd.get((int)InputParam::Width, 0);
    shape = {n, c, h, w};
    return 0;
}

int InputLayer::forward_inplace(Tensor &bottom_top_blob, const NetOption &opt) const {
    return 0;
}

REGISTER_LAYER_CLASS(Input);

}   // end namespace otter
