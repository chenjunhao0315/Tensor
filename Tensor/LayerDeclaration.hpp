//
//  LayerDeclaration.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/4/8.
//

#ifndef LayerDeclaration_h
#define LayerDeclaration_h

#include "LayerRegistry.hpp"
#include "InputLayer.hpp"
#include "ConvolutionLayer.hpp"
#include "DeconvolutionLayer.hpp"
#include "BatchNormalizationLayer.hpp"
#include "LReluLayer.hpp"
#include "ReluLayer.hpp"
#include "Relu6Layer.hpp"
#include "ShortCutLayer.hpp"
#include "EltwiseLayer.hpp"
#include "SplitLayer.hpp"
#include "MaxPoolLayer.hpp"
#include "DropoutLayer.hpp"
#include "ConcatLayer.hpp"
#include "UpsampleLayer.hpp"
#include "ChannelShuffleLayer.hpp"
#include "CropLayer.hpp"
#include "SliceLayer.hpp"
#include "ReshapeLayer.hpp"
#include "PermuteLayer.hpp"
#include "SigmoidLayer.hpp"
#include "Yolov3DetectionOutputLayer.hpp"
#include "NanodetPlusDetectionOutputLayer.hpp"

namespace otter {

REGISTER_LAYER_CLASS(Input);
REGISTER_LAYER_CLASS(Convolution);
REGISTER_LAYER_CLASS(Deconvolution);
REGISTER_LAYER_CLASS(BatchNormalization);
REGISTER_LAYER_CLASS(LRelu);
REGISTER_LAYER_CLASS(Relu);
REGISTER_LAYER_CLASS(Relu6);
REGISTER_LAYER_CLASS(ShortCut);
REGISTER_LAYER_CLASS(Eltwise);
REGISTER_LAYER_CLASS(Split);
REGISTER_LAYER_CLASS(MaxPool);
REGISTER_LAYER_CLASS(Dropout);
REGISTER_LAYER_CLASS(Concat);
REGISTER_LAYER_CLASS(Upsample);
REGISTER_LAYER_CLASS(ChannelShuffle);
REGISTER_LAYER_CLASS(Crop);
REGISTER_LAYER_CLASS(Slice);
REGISTER_LAYER_CLASS(Reshape);
REGISTER_LAYER_CLASS(Permute);
REGISTER_LAYER_CLASS(Sigmoid);
REGISTER_LAYER_CLASS(Yolov3DetectionOutput);
REGISTER_LAYER_CLASS(NanodetPlusDetectionOutput);

}   // end namespace otter

#endif /* LayerDeclaration_h */
