//
//  ParamDict.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/2/19.
//

#include "ParamDict.hpp"

namespace otter {

ParamDict::ParamDict() {
    clear();
}

ParamDict::~ParamDict() {
    
}

ParamType ParamDict::type(int id) const {
    return params[id].type;
}

int ParamDict::get(int id, int def) const {
    return (params[id].type != ParamType::Undefined) ? params[id].integer : def;
}

float ParamDict::get(int id, float def) const {
    return (params[id].type != ParamType::Undefined) ? params[id].floating : def;
}

Tensor ParamDict::get(int id, const Tensor& def) const {
    return (params[id].type != ParamType::Undefined) ? params[id].t : def;
}

void ParamDict::set(int id, int i) {
    params[id].type = ParamType::Int;
    params[id].integer = i;
}

void ParamDict::set(int id, float f) {
    params[id].type = ParamType::Float;
    params[id].floating = f;
}

void ParamDict::set(int id, const Tensor& t) {
    params[id].type = ParamType::ArrayIntFloat;
    params[id].t = t;
}

void ParamDict::clear() {
    for (const auto i : otter::irange(MAX_PARAM_COUNT)) {
        params[i].type = ParamType::Undefined;
        params[i].t = Tensor();
    }
}

}
