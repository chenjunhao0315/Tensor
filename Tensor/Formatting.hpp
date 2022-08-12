//
//  Formatting.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/2/11.
//

#ifndef Formatting_hpp
#define Formatting_hpp

#include <ostream>
#include "Tensor.hpp"

namespace otter {

static bool summary = true;
static int64_t edgeitems = 3;

static void set_summary(bool _summary) {
    summary = _summary;
}

static void set_print_edgeitems(int64_t items) {
    edgeitems = items;
}

void print(const Tensor & t, int64_t linesize = 80);
std::ostream& print(std::ostream& out, const Tensor& t, int64_t linesize);

void set_summary(bool summary);
void set_print_edgeitems(int64_t items);

std::ostream& tensor_str(std::ostream& stream, const Tensor& tensor);
void _tensor_str(const Tensor& tensor, std::ostream& stream, double scale, int64_t sz, int64_t indent, bool head);
void vector_str(const Tensor& tensor, std::ostream& stream, double scale, int64_t sz, int64_t indent, bool head);
void _vector_str(const Tensor& tensor, std::ostream& stream, double scale, int64_t sz);
void scalar_str(const Tensor& tensor, std::ostream& stream);

static inline std::ostream& operator<<(std::ostream& out, const Tensor& t) {
    if (t.elempack() > 1) {
        t.print();
        if (summary)
            return tensor_str(out, t.packing(1));
        return print(out, t.packing(1), 80);
    }
    if (summary)
        return tensor_str(out, t);
    return print(out, t, 80);
}

std::ostream& operator<<(std::ostream& out, Scalar s);

}   // end namespace otter

#endif /* Formatting_hpp */
