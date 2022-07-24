//
//  tensor_str.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/7/14.
//

#ifndef TENSOR_STR_HPP
#define TENSOR_STR_HPP

#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <tuple>
#include <string>

#include <Tensor.hpp>

namespace otter {

bool summary = true;
int64_t edgeitems = 3;

inline std::ios_base& defaultfloat(std::ios_base& __base) {
    __base.unsetf(std::ios_base::floatfield);
    return __base;
}
//saves/restores number formatting inside scope
struct FormatGuard {
    FormatGuard(std::stringstream & out)
    : out(out), saved(nullptr) {
        saved.copyfmt(out);
    }
    ~FormatGuard() {
        out.copyfmt(saved);
    }
private:
    std::stringstream & out;
    std::ios saved;
};

static std::tuple<double, int64_t> __printFormat(std::stringstream& stream, const Tensor& self) {
    auto size = self.numel();
    if(size == 0) {
        return std::make_tuple(1., 0);
    }
    bool intMode = true;
    auto self_p = self.data_ptr<double>();
    for (const auto i : otter::irange(size)) {
        auto z = self_p[i];
        if(std::isfinite(z)) {
            if(z != std::ceil(z)) {
                intMode = false;
                break;
            }
        }
    }
    int64_t offset = 0;
    while(!std::isfinite(self_p[offset])) {
        offset = offset + 1;
        if(offset == size) {
            break;
        }
    }
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    double expMin;
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    double expMax;
    if(offset == size) {
        expMin = 1;
        expMax = 1;
    } else {
        expMin = fabs(self_p[offset]);
        expMax = fabs(self_p[offset]);
        for (const auto i : otter::irange(offset, size)) {
            double z = fabs(self_p[i]);
            if(std::isfinite(z)) {
                if(z < expMin) {
                    expMin = z;
                }
                if(self_p[i] > expMax) {
                    expMax = z;
                }
            }
        }
        if(expMin != 0) {
            expMin = std::floor(std::log10(expMin)) + 1;
        } else {
            expMin = 1;
        }
        if(expMax != 0) {
            expMax = std::floor(std::log10(expMax)) + 1;
        } else {
            expMax = 1;
        }
    }
    double scale = 1;
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    int64_t sz;
    if(intMode) {
        if(expMax > 9) {
            sz = 11;
            stream << std::scientific << std::setprecision(4);
        } else {
            sz = expMax + 1;
            stream << defaultfloat;
        }
    } else {
        if(expMax-expMin > 4) {
            sz = 11;
            if(std::fabs(expMax) > 99 || std::fabs(expMin) > 99) {
                sz = sz + 1;
            }
            stream << std::scientific << std::setprecision(4);
        } else {
            if(expMax > 5 || expMax < 0) {
                sz = 7;
                scale = std::pow(10, expMax-1);
                stream << std::fixed << std::setprecision(4);
            } else {
                if(expMax == 0) {
                    sz = 7;
                } else {
                    sz = expMax+6;
                }
                stream << std::fixed << std::setprecision(4);
            }
        }
    }
    return std::make_tuple(scale, sz);
}

static void __printIndent(std::stringstream &stream, int64_t indent) {
    for (const auto i : otter::irange(indent)) {
        (void)i; //Suppress unused variable warning
        stream << " ";
    }
}

static void printScale(std::stringstream & stream, double scale) {
    FormatGuard guard(stream);
    stream << defaultfloat << scale << " *" << std::endl;
}

std::string scalar_str(const Tensor& tensor) {
    std::stringstream stream;
    
    stream << defaultfloat << tensor.data_ptr<double>()[0] << std::endl;
    
    std::string str = "[" + stream.str() + "]";
    
    return str;
}

std::string _vector_str(const Tensor& tensor, double scale, int64_t sz) {
    std::stringstream stream;
    
    double* tensor_p = tensor.data_ptr<double>();
    for (const auto i : otter::irange(tensor.size(0))) {
        if (i)
            stream << ", ";
        stream << std::setw((int)sz) << tensor_p[i] / scale;
    }
    
    return stream.str();
}

std::string vector_str(const Tensor& tensor) {
    std::stringstream stream;
    
    double scale;
    int64_t sz;
    std::tie(scale, sz) =  __printFormat(stream, tensor);
    
    if(scale != 1) {
        printScale(stream, scale);
    }
    
    std::string str = stream.str() + "[";
    if (summary && tensor.size(0) > 2 * edgeitems) {
        str += _vector_str(tensor.slice(0, 0, edgeitems, 1), scale, sz);
        str += " ... ";
        str += _vector_str(tensor.slice(0, -edgeitems, tensor.size(0), 1), scale, sz);
    } else {
        str += _vector_str(tensor, scale, sz);
    }
    str += "]";
    
    return str;
}

std::string tensor_str(const Tensor& tensor) {
    int64_t dim = tensor.dim();
    
    std::string str = "[";
    
    if (dim == 0) {
        return scalar_str(tensor);
    }
    
    if (dim == 1) {
        return vector_str(tensor);
    }
    
    if (summary && tensor.size(0) > 2 * edgeitems) {
        for (const auto i : otter::irange(0, edgeitems)) {
            str += tensor_str(tensor[i]);
            str += ",\n";
            if (dim >= 3)
                str += "\n";
        }
        str += ".\n.\n.\n";
        if (dim >= 3)
            str += "\n";
        for (const auto i : otter::irange(tensor.size(0) - edgeitems, tensor.size(0))) {
            str += tensor_str(tensor[i]);
            if (i != tensor.size(0) - 1) {
                str += ",\n";
                if (dim >= 3)
                    str += "\n";
            }
        }
    } else {
        for (const auto i : otter::irange(0, tensor.size(0))) {
            str += tensor_str(tensor[i]);
            if (i != tensor.size(0) - 1) {
                str += ",\n";
                if (dim >= 3)
                    str += "\n";
            }
        }
    }
    str += "]";
    
    return str;
}

}   // end namespace otter

#endif  // TENSOR_STR_HPP
