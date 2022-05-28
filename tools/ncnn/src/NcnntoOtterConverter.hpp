//
//  OtterConverter.hpp
//  Otter
//
//  Created by 陳均豪 on 2022/03/10.
//

#ifndef OtterConverter_hpp
#define OtterConverter_hpp

#include "Otter.hpp"
#include "datareader.h"

using namespace otter::core;

namespace otter {

OtterLeader ncnn2otter(const char *model_path);


}

#endif /* OtterConverter_hpp */
