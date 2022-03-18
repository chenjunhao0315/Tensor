//
//  CPUGenerator.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/3/18.
//

#ifndef CPUGenerator_hpp
#define CPUGenerator_hpp

#include "GeneratorNucleus.hpp"
#include "MT19937.hpp"

namespace otter {

namespace detail {

const Generator& getDefaultCPUGenerator();

Generator createCPUGenerator(uint64_t seed_val);

}   // end namespace detail

struct CPUGeneratorNucleus : public GeneratorNucleus {
    CPUGeneratorNucleus(uint64_t seed_in = default_rng_seed_val);
    ~CPUGeneratorNucleus() override = default;
    
    std::shared_ptr<CPUGeneratorNucleus> clone() const;
    void set_current_seed(uint64_t seed) override;
    uint64_t current_seed() const override;
    uint64_t seed() override;
//    void set_state(const c10::TensorImpl& new_state) override;
//    c10::intrusive_ptr<c10::TensorImpl> get_state() const override;
//    static DeviceType device_type();
    uint32_t random();
    uint64_t random64();
//    c10::optional<float> next_float_normal_sample();
//    c10::optional<double> next_double_normal_sample();
//    void set_next_float_normal_sample(c10::optional<float> randn);
//    void set_next_double_normal_sample(c10::optional<double> randn);
    otter::mt19937 engine();
    void set_engine(otter::mt19937 engine);

private:
    CPUGeneratorNucleus* clone_impl() const override;
    otter::mt19937 engine_;
//    c10::optional<float> next_float_normal_sample_;
//    c10::optional<double> next_double_normal_sample_;
};


}   // end namespace otter

#endif /* CPUGenerator_hpp */
