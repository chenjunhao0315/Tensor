//
//  OTensor.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/2/9.
//

#ifndef OTensor_h
#define OTensor_h

// Core
#include "Macro.hpp"
#include "Tensor.hpp"
#include "TensorBase.hpp"
#include "TensorAccessor.hpp"
#include "TensorFactory.hpp"
#include "TensorMaker.hpp"
#include "TensorOperator.hpp"
#include "Formatting.hpp"
#include "TensorOptions.hpp"
#include "TensorIterator.hpp"

// Memory
#include "Device.hpp"
#include "RefPtr.hpp"
#include "Allocator.hpp"
#include "CPUAllocator.hpp"
#include "CPUCachingAllocator.hpp"
#include "CPUProfilingAllocator.hpp"
#include "Memory.hpp"
#include "MemoryFormat.hpp"

// Dtype
#include "ScalarType.hpp"
#include "TypeCast.hpp"
#include "PackedData.hpp"

// Utils
#include "AutoBuffer.hpp"
#include "Utils.hpp"
#include "Accumulator.hpp"
#include "Clock.hpp"
#include "ArrayRef.hpp"
#include "SmallVector.hpp"
#include "flat_hash_map.hpp"
#include "Exception.hpp"
#include "StringUtils.hpp"
#include "Math.hpp"

// Parallel
#include "Parallel.hpp"

// Convolution
#include "Convolution.hpp"
#include "ConvolutionUtils.hpp"
#include "im2col.hpp"

// Normalization
#include "Normalization.hpp"

// Distribution
#include "TensorDistribution.hpp"

// Ops
#include "TensorFunction.hpp"
#include "TensorShape.hpp"
#include "TensorBlas.hpp"
#include "Dispatch.hpp"
#include "DispatchStub.hpp"
#include "TensorLinearAlgebra.hpp"
#include "Activation.hpp"
#include "Pool.hpp"
#include "Padding.hpp"
#include "Dropout.hpp"
#include "TensorInterpolation.hpp"
#include "ChannelShuffle.hpp"
#include "TensorTransform.hpp"
#include "TensorCompare.hpp"
#include "TensorSoftmax.hpp"
#include "Quantize.hpp"
#include "TensorPacking.hpp"
#include "TensorEltwise.hpp"

// Net
#include "ParamDict.hpp"
#include "NetOption.hpp"
#include "Net.hpp"
#include "Benchmark.hpp"
#include "Blob.hpp"
#include "DataReader.hpp"

// Layer
#include "Layer.hpp"
#include "LayerRegistry.hpp"
#include "Initializer.hpp"

// cv
#include "Vision.hpp"
#include "TensorPixel.hpp"
#include "Drawing.hpp"
#include "GraphicAPI.hpp"
#include "DefaultColor.hpp"
#include "ColorConvert.hpp"
#include "ImageThreshold.hpp"
#include "DrawDetection.hpp"
#include "LineDetection.hpp"
#include "KalmanFilter.hpp"
#include "KalmanTracker.hpp"
#include "Hungarian.hpp"
#include "Stabilizer.hpp"
#include "PoseEstimation.hpp"

#endif /* OTensor_h */
