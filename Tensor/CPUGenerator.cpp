//
//  CPUGenerator.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/3/18.
//

#include "Generator.hpp"
#include "CPUGenerator.hpp"

namespace otter {

namespace detail {

Generator createCPUGenerator(uint64_t seed_val) {
    return make_generator<CPUGeneratorNucleus>(seed_val);
}

const Generator& getDefaultCPUGenerator() {
    static auto default_gen_cpu = createCPUGenerator(otter::detail::getNonDeterministicRandom());
    return default_gen_cpu;
}

inline uint64_t make64BitsFrom32Bits(uint32_t hi, uint32_t lo) {
    return (static_cast<uint64_t>(hi) << 32) | lo;
}

}   // end namespace detail

CPUGeneratorNucleus::CPUGeneratorNucleus(uint64_t seed_in) : otter::GeneratorNucleus(Device::CPU), engine_{seed_in} {}

void CPUGeneratorNucleus::set_current_seed(uint64_t seed) {
    engine_ = mt19937(seed);
}

uint64_t CPUGeneratorNucleus::current_seed() const {
    return engine_.seed();
}

uint64_t CPUGeneratorNucleus::seed() {
    auto random = otter::detail::getNonDeterministicRandom();
    this->set_current_seed(random);
    return random;
}

uint32_t CPUGeneratorNucleus::random() {
    return engine_();
}

uint64_t CPUGeneratorNucleus::random64() {
    uint32_t random1 = engine_();
    uint32_t random2 = engine_();
    return detail::make64BitsFrom32Bits(random1, random2);
}

otter::mt19937 CPUGeneratorNucleus::engine() {
    return engine_;
}


void CPUGeneratorNucleus::set_engine(otter::mt19937 engine) {
    engine_ = engine;
}

std::shared_ptr<CPUGeneratorNucleus> CPUGeneratorNucleus::clone() const {
    return std::shared_ptr<CPUGeneratorNucleus>(this->clone_impl());
}

CPUGeneratorNucleus* CPUGeneratorNucleus::clone_impl() const {
    auto gen = new CPUGeneratorNucleus();
    gen->set_engine(engine_);
    //    gen->set_next_float_normal_sample(next_float_normal_sample_);
    //    gen->set_next_double_normal_sample(next_double_normal_sample_);
    return gen;
}

}   // end namespace otter
