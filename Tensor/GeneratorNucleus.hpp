//
//  GeneratorNucleus.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/3/18.
//

#ifndef GeneratorNucleus_hpp
#define GeneratorNucleus_hpp

#include <chrono>
#include <mutex>

#include "Device.hpp"
#include "RefPtr.hpp"

namespace otter {

constexpr uint64_t default_rng_seed_val = 67280421310721;

struct GeneratorNucleus : public Ptr_quantum {
    GeneratorNucleus(Device device);
    
    GeneratorNucleus(const GeneratorNucleus& other) = delete;
    GeneratorNucleus(GeneratorNucleus&& other) = delete;
    GeneratorNucleus& operator=(const GeneratorNucleus& other) = delete;
    
    virtual ~GeneratorNucleus() = default;
    Ptr<GeneratorNucleus> clone() const;
    
    virtual void set_current_seed(uint64_t seed) = 0;
    virtual uint64_t current_seed() const = 0;
    virtual uint64_t seed() = 0;
    //    virtual void set_state(const TensorNucleus& new_state) = 0;
    //    virtual Ptr<TensorNucleus> get_state() const = 0;
    Device device() const;
    
    // See Note [Acquire lock when using random generators]
    std::mutex mutex_;
    
protected:
    Device device_;
    
    virtual GeneratorNucleus* clone_impl() const = 0;
};

namespace detail {

uint64_t getNonDeterministicRandom(bool is_cuda = false);

} // namespace detail

}   // end namespace otter

#endif /* GeneratorNucleus_hpp */
