//
//  NetOption.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/2/19.
//

#include "NetOption.hpp"

namespace otter {

NetOption::NetOption() {
    lightmode = true;
    train = false;
    use_non_lib_optimize = true;
    use_packing_layout = true;
    openmp_blocktime = 20;
}

}
