//
//  NetOption.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/2/19.
//

#ifndef NetOption_hpp
#define NetOption_hpp

namespace otter {

class NetOption {
public:
    NetOption();
    
public:
    bool lightmode;
    bool train;
    bool use_non_lib_optimize;
};

enum class CompileMode {
    Inference,
    Initial
};

}

#endif /* NetOption_hpp */
