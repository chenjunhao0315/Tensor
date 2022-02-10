//
//  Formatting.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/2/11.
//

#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <tuple>

#include "Utils.hpp"
#include "Formatting.hpp"

namespace otter {

inline std::ios_base& defaultfloat(std::ios_base& __base) {
    __base.unsetf(std::ios_base::floatfield);
    return __base;
}

static std::tuple<double, int64_t> __printFormat(std::ostream& stream, const Tensor& self) {
    auto size = self.numel();
    if(size == 0) {
        return std::make_tuple(1., 0);
    }
    bool intMode = true;
    auto self_p = self.data<double>();
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

static void __printIndent(std::ostream &stream, int64_t indent)
{
    for (const auto i : otter::irange(indent)) {
        (void)i; //Suppress unused variable warning
        stream << " ";
    }
}

static void printScale(std::ostream & stream, double scale) {
    //    FormatGuard guard(stream);
    stream << defaultfloat << scale << " *" << std::endl;
}
static void __printMatrix(std::ostream& stream, const Tensor& self, int64_t linesize, int64_t indent)
{
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    double scale;
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    int64_t sz;
    std::tie(scale, sz) = __printFormat(stream, self);
    
    __printIndent(stream, indent);
    int64_t nColumnPerLine = (linesize-indent)/(sz+1);
    int64_t firstColumn = 0;
    int64_t lastColumn = -1;
    while(firstColumn < self.size(1)) {
        if(firstColumn + nColumnPerLine <= self.size(1)) {
            lastColumn = firstColumn + nColumnPerLine - 1;
        } else {
            lastColumn = self.size(1) - 1;
        }
        if(nColumnPerLine < self.size(1)) {
            if(firstColumn != 0) {
                stream << std::endl;
            }
            stream << "Columns " << firstColumn+1 << " to " << lastColumn+1;
            __printIndent(stream, indent);
        }
        if(scale != 1) {
            printScale(stream,scale);
            __printIndent(stream, indent);
        }
        for (const auto l : otter::irange(self.size(0))) {
            Tensor row = self.select(0, l);
            double *row_ptr = row.data<double>();
            for (const auto c : otter::irange(firstColumn, lastColumn+1)) {
                stream << std::setw(sz) << row_ptr[c]/scale;
                if(c == lastColumn) {
                    stream << std::endl;
                    if(l != self.size(0)-1) {
                        if(scale != 1) {
                            __printIndent(stream, indent);
                            stream << " ";
                        } else {
                            __printIndent(stream, indent);
                        }
                    }
                } else {
                    stream << " ";
                }
            }
        }
        firstColumn = lastColumn + 1;
    }
}

void __printTensor(std::ostream& stream, Tensor& self, int64_t linesize)
{
    std::vector<int64_t> counter(self.dim() - 2);
    bool start = true;
    bool finished = false;
    counter[0] = -1;
    for (const auto i : otter::irange(1, counter.size())) {
        counter[i] = 0;
    }
    while(true) {
        for(int64_t i = 0; self.dim() - 2; i++) {
            counter[i] = counter[i] + 1;
            if(counter[i] >= self.size(i)) {
                if(i == self.dim() - 3) {
                    finished = true;
                    break;
                }
                counter[i] = 0;
            } else {
                break;
            }
        }
        if(finished) {
            break;
        }
        if(start) {
            start = false;
        } else {
            stream << std::endl;
        }
        stream << "(";
        Tensor tensor = self;
        for (const auto i : otter::irange(self.dim() - 2)) {
            tensor = tensor.select(0, counter[i]);
            stream << counter[i]+1 << ",";
        }
        stream << ".,.) = " << std::endl;
        __printMatrix(stream, tensor, linesize, 1);
    }
}

void print(const Tensor & t, int64_t linesize) {
    print(std::cout,t,linesize);
}

std::ostream& print(std::ostream& stream, const Tensor& tensor_, int64_t linesize) {
    // Maybe ostream guard
    if (!tensor_.defined()) {
        stream << "[Tensor] Undefined!";
    } else {
        Tensor tensor = tensor_.to(ScalarType::Double);
        if (tensor.dim() == 0) {
            stream << defaultfloat << tensor.data<double>()[0] << std::endl;
            stream << "[ " << tensor_.toString() << "{}";
        } else if (tensor.dim() == 1) {
            if (tensor.numel() > 0) {
                double scale;
                int64_t sz;
                
                std::tie(scale, sz) =  __printFormat(stream, tensor);
                if(scale != 1) {
                    printScale(stream, scale);
                }
                double* tensor_p = tensor.data<double>();
                for (const auto i : otter::irange(tensor.size(0))) {
                    stream << std::setw(sz) << tensor_p[i] / scale << std::endl;
                }
            }
            stream << "[ " << tensor_.toString() << "{" << tensor.size(0) << "}";
        } else if(tensor.dim() == 2) {
            if (tensor.numel() > 0) {
                __printMatrix(stream, tensor, linesize, 0);
            }
            stream << "[ " << tensor_.toString() << "{" << tensor.size(0) << "," <<  tensor.size(1) << "}";
        } else {
            if (tensor.numel() > 0) {
                __printTensor(stream, tensor, linesize);
            }
            stream << "[ " << tensor_.toString() << "{" << tensor.size(0);
            for (const auto i : otter::irange(1, tensor.dim())) {
                stream << "," << tensor.size(i);
            }
            stream << "}";
        }
        
        stream << " ]";
    }
    
    return stream;
}



}
