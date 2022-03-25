//
//  TensorDistributionKernel.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/3/19.
//

#include "TensorDistribution.hpp"
#include "TensorDistributionTemplate.hpp"
#include "DistributionsHelper.hpp"
#include "CPUGenerator.hpp"

#include "UnaryOps.hpp"

namespace otter {
namespace native {
namespace {

void uniform_kernel(TensorIterator& iter, double from, double to, Generator gen) {
    CPUGeneratorNucleus* generator = get_generator_or_default<CPUGeneratorNucleus>(gen, detail::getDefaultCPUGenerator());
    templates::cpu::uniform_kernel(iter, from, to, generator);
}

void normal_kernel(const TensorBase &self, double mean, double std, Generator gen) {
    CPUGeneratorNucleus* generator = get_generator_or_default<CPUGeneratorNucleus>(gen, detail::getDefaultCPUGenerator());
    templates::cpu::normal_kernel(self, mean, std, generator);
}

static void random_from_to_kernel(TensorIterator& iter, uint64_t range, int64_t base, Generator gen) {
    CPUGeneratorNucleus* generator = get_generator_or_default<CPUGeneratorNucleus>(gen, detail::getDefaultCPUGenerator());
    templates::cpu::random_from_to_kernel(iter, range, base, generator);
}
static void random_kernel(TensorIterator& iter, Generator gen) {
    CPUGeneratorNucleus* generator = get_generator_or_default<CPUGeneratorNucleus>(gen, detail::getDefaultCPUGenerator());
    templates::cpu::random_kernel(iter, generator);
}

static void random_full_64_bits_range_kernel(TensorIterator& iter, Generator gen) {
    CPUGeneratorNucleus* generator = get_generator_or_default<CPUGeneratorNucleus>(gen, detail::getDefaultCPUGenerator());
    templates::cpu::random_full_64_bits_range_kernel(iter, generator);
}

}   // end namespace
}   // end namespace native

REGISTER_DISPATCH(uniform_stub, &native::uniform_kernel);
REGISTER_DISPATCH(normal_stub, &native::normal_kernel);
REGISTER_DISPATCH(random_from_to_stub, &native::random_from_to_kernel);
REGISTER_DISPATCH(random_full_64_bits_range_stub, &native::random_full_64_bits_range_kernel);
REGISTER_DISPATCH(random_stub, &native::random_kernel);

}   // end namespace otter
