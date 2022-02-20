//
//  ParamDict.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/2/19.
//

#ifndef ParamDict_hpp
#define ParamDict_hpp

#include "Tensor.hpp"

namespace otter {

#define MAX_PARAM_COUNT 32

enum class ParamType {
    Undefined,
    IntFloat,
    Int,
    Float,
    ArrayIntFloat,
    ArrayInt,
    ArrayFloat
};

class ParamDict {
public:
    ParamDict();
    virtual ~ParamDict();

    // copy
    ParamDict(const ParamDict&);
    // assign
    ParamDict& operator=(const ParamDict&);
    
    // get type
    ParamType type(int id) const;

    // get int
    int get(int id, int def) const;
    // get float
    float get(int id, float def) const;
    // get array
    Tensor get(int id, const Tensor& def) const;

    // set int
    void set(int id, int i);
    // set float
    void set(int id, float f);
    // set array
    void set(int id, const Tensor& v);

    void clear();
    
private:
    struct {
        ParamType type;
        union {
            int     integer;
            float   floating;
        };
        Tensor t;
    } params[MAX_PARAM_COUNT];
};

}   // end namespace otter

#endif /* ParamDict_hpp */
