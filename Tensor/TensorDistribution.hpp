//
//  TensorDistribution.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/3/18.
//

#ifndef TensorDistribution_hpp
#define TensorDistribution_hpp

#include <cstdint>

namespace otter {
class Tensor;
struct Generator;

namespace native {

Tensor& uniform_(Tensor& self, double from, double to);
Tensor& uniform_(Tensor& self, double from, double to, Generator gen);

Tensor& normal_(Tensor& self, double mean, double std);
Tensor& normal_(Tensor& self, double mean, double std, Generator gen);

Tensor& normal_out(const Tensor& mean, double std, Generator gen, Tensor& output);
Tensor& normal_out(double mean, const Tensor& std, Generator gen, Tensor& output);
Tensor& normal_out(const Tensor& mean, const Tensor& std, Generator gen, Tensor& output);

Tensor normal(const Tensor& mean, double std, Generator gen);
Tensor normal(double mean, const Tensor& std, Generator gen);
Tensor normal(const Tensor& mean, const Tensor& std, Generator gen);

Tensor& random_(Tensor& self);
Tensor& random_(Tensor& self, Generator gen);

Tensor& random_(Tensor& self, int64_t from, int64_t to);
Tensor& random_(Tensor& self, int64_t from, int64_t to, Generator gen);

Tensor& random_(Tensor& self, int64_t to);
Tensor& random_(Tensor& self, int64_t to, Generator gen);

}   // end namespace native
}   // end namespace otter

#endif /* TensorDistribution_hpp */
