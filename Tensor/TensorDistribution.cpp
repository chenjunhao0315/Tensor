//
//  TensorDistribution.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/3/18.
//

#include "TensorDistribution.hpp"

#include "Tensor.hpp"
#include "UnaryOps.hpp"
#include "TensorDistributionTemplate.hpp"

namespace otter {
namespace native {

DEFINE_DISPATCH(uniform_stub);
DEFINE_DISPATCH(normal_stub);
DEFINE_DISPATCH(random_from_to_stub);
DEFINE_DISPATCH(random_full_64_bits_range_stub);
DEFINE_DISPATCH(random_stub);


template<typename RNG>
struct UniformStub {
    void operator()(TensorIterator& iter, double from, double to, Generator gen) {
        uniform_stub(Device::CPU, iter, from, to, gen);
    }
};

template<typename RNG>
struct UniformMeta {
    // No-op!
    void operator()(TensorIterator& /*iter*/, double /*from*/, double /*to*/, Generator /*gen*/) {
    }
};

Tensor& uniform_(Tensor& self, double from, double to) {
    return otter::native::templates::uniform_impl_<UniformStub, Generator>(self, from, to, Generator());
}

Tensor& uniform_(Tensor& self, double from, double to, Generator gen) {
    return otter::native::templates::uniform_impl_<UniformStub, Generator>(self, from, to, gen);
}

Tensor& uniform_meta_(Tensor& self, double from, double to, Generator gen) {
    return otter::native::templates::uniform_impl_<UniformMeta, Generator>(self, from, to, gen);
}

template<typename RNG>
struct NormalStub {
    void operator()(Tensor& self, double mean, double std, Generator gen) {
        normal_stub(Device::CPU, self, mean, std, gen);
    }
};

template<typename RNG>
struct NormalMeta {
    // No-op!
    void operator()(Tensor& /*self*/, double /*mean*/, double /*std*/, Generator /*gen*/) {
    }
};

// inplace
Tensor& normal_(Tensor& self, double mean, double std) {
    return otter::native::templates::normal_impl_<NormalStub, Generator>(self, mean, std, Generator());
}

Tensor& normal_(Tensor& self, double mean, double std, Generator gen) {
    return otter::native::templates::normal_impl_<NormalStub, Generator>(self, mean, std, gen);
}

Tensor& normal_meta_(Tensor& self, double mean, double std, Generator gen) {
    return otter::native::templates::normal_impl_<NormalMeta, Generator>(self, mean, std, gen);
}

// out tensor float
Tensor& normal_out(const Tensor& mean, double std, Generator gen, Tensor& output) {
    return otter::native::templates::normal_out_impl<NormalStub, Generator>(output, mean, std, gen);
}

Tensor& normal_out_meta(const Tensor& mean, double std, Generator gen, Tensor& output) {
    return otter::native::templates::normal_out_impl<NormalMeta, Generator>(output, mean, std, gen);
}

// out float tensor
Tensor& normal_out(double mean, const Tensor& std, Generator gen, Tensor& output) {
    return otter::native::templates::normal_out_impl<NormalStub, Generator>(output, mean, std, gen);
}

Tensor& normal_out_meta(double mean, const Tensor& std, Generator gen, Tensor& output) {
    return otter::native::templates::normal_out_impl<NormalMeta, Generator>(output, mean, std, gen);
    
}

// out tensor tensor
Tensor& normal_out(const Tensor& mean, const Tensor& std, Generator gen, Tensor& output) {
    return otter::native::templates::normal_out_impl<NormalStub, Generator>(output, mean, std, gen);
}

Tensor& normal_out_meta(const Tensor& mean, const Tensor& std, Generator gen, Tensor& output) {
    return otter::native::templates::normal_out_impl<NormalMeta, Generator>(output, mean, std, gen);
}

// functional tensor float
Tensor normal(const Tensor& mean, double std, Generator gen) {
    return otter::native::templates::normal_impl<NormalStub, Generator>(mean, std, gen);
}

Tensor normal_meta(const Tensor& mean, double std, Generator gen) {
    return otter::native::templates::normal_impl<NormalMeta, Generator>(mean, std, gen);
}

// functional float tensor
Tensor normal(double mean, const Tensor& std, Generator gen) {
    return otter::native::templates::normal_impl<NormalStub, Generator>(mean, std, gen);
}

Tensor normal_meta(double mean, const Tensor& std, Generator gen) {
    return otter::native::templates::normal_impl<NormalMeta, Generator>(mean, std, gen);
}

// functional tensor tensor
Tensor normal(const Tensor& mean, const Tensor& std, Generator gen) {
    return otter::native::templates::normal_impl<NormalStub, Generator>(mean, std, gen);
}

Tensor normal_meta(const Tensor& mean, const Tensor& std, Generator gen) {
    return otter::native::templates::normal_impl<NormalMeta, Generator>(mean, std, gen);
}

template<typename RNG>
struct RandomStub {
    void operator()(TensorIterator& iter, Generator gen) {
        random_stub(Device::CPU, iter, gen);
    }
};

Tensor& random_(Tensor& self) {
    return otter::native::templates::random_impl<RandomStub, Generator>(self, Generator());
}

Tensor& random_(Tensor& self, Generator gen) {
    return otter::native::templates::random_impl<RandomStub, Generator>(self, gen);
}

template<typename RNG>
struct RandomFromToStub {
    void operator()(TensorIterator& iter, uint64_t range, int64_t from, Generator gen) {
        random_from_to_stub(Device::CPU, iter, range, from, gen);
    }
    void operator()(TensorIterator& iter, Generator gen) {
        random_full_64_bits_range_stub(Device::CPU, iter, gen);
    }
};

template<typename RNG>
struct RandomFromToMeta {
    // No-op!
    void operator()(TensorIterator& /*iter*/, uint64_t /*range*/, int64_t /*from*/, Generator /*gen*/) {
    }
    void operator()(TensorIterator& /*iter*/, Generator /*gen*/) {
    }
};

Tensor& random_(Tensor& self, int64_t from, int64_t to) {
    return otter::native::templates::random_from_to_impl<RandomFromToStub, Generator>(self, from, to, Generator());
}

Tensor& random_(Tensor& self, int64_t from, int64_t to, Generator gen) {
    return otter::native::templates::random_from_to_impl<RandomFromToStub, Generator>(self, from, to, gen);
}

Tensor& random_(Tensor& self, int64_t to) {
    return random_(self, 0, to);
}

Tensor& random_(Tensor& self, int64_t to, Generator gen) {
    return random_(self, 0, to, gen);
}

Tensor& random_meta_(Tensor& self, Generator /*gen*/) {
    // No error checking yay
    return self;
}

Tensor& random_meta_(Tensor& self, int64_t from, int64_t to, Generator gen) {
    return otter::native::templates::random_from_to_impl<RandomFromToMeta, Generator>(self, from, to, gen);
}

Tensor& random_meta_(Tensor& self, int64_t to, Generator gen) {
    return random_meta_(self, 0, to, gen);
}

}   // end namespace native
}   // end namespace otter
