//
//  LayerDeclaration.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/2/19.
//

#ifndef LayerDeclaration_hpp
#define LayerDeclaration_hpp

#include "InputLayer.hpp"
#include "ConvolutionLayer.hpp"

namespace otter {

enum class LayerType {
    Input,
    Convolution
};

}

#endif /* LayerDeclaration_hpp */
