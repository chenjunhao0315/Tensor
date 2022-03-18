//
//  Generator.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/3/18.
//

#ifndef Generator_hpp
#define Generator_hpp

#include "Exception.hpp"
#include "GeneratorNucleus.hpp"

namespace otter {

struct Generator {
    Generator() {}
    
    explicit Generator(Ptr<otter::GeneratorNucleus> gen_impl)
    : nucleus_(std::move(gen_impl)) {
        if (nucleus_.get() == nullptr) {
            throw std::runtime_error("GeneratorNucleus with nullptr is not supported");
        }
    }
    
    bool operator==(const Generator& rhs) const {
        return this->nucleus_ == rhs.nucleus_;
    }
    
    bool operator!=(const Generator& rhs) const {
        return !((*this) == rhs);
    }
    
    bool defined() const {
        return static_cast<bool>(nucleus_);
    }
    
    otter::GeneratorNucleus* unsafeGetGeneratorNucleus() const {
        return nucleus_.get();
    }
    
    otter::GeneratorNucleus* unsafeReleaseGeneratorNucleus() {
        return nucleus_.release();
    }
    
    const otter::Ptr<otter::GeneratorNucleus>& getPtr() const {
        return nucleus_;
    }
    
    void set_current_seed(uint64_t seed) { nucleus_->set_current_seed(seed); }
    
    uint64_t current_seed() const { return nucleus_->current_seed(); }
    
    uint64_t seed() { return nucleus_->seed(); }
    
//    void set_state(const otter::Tensor& new_state);
//
//    otter::Tensor get_state() const;
    
    std::mutex& mutex() {
        return nucleus_->mutex_;
    }
    
    Device device() const { return nucleus_->device(); }
    
    template<typename T>
    T* get() const { return static_cast<T*>(nucleus_.get()); }
    
    Generator clone() const {
        return Generator(nucleus_->clone());
    }
    
private:
    Ptr<GeneratorNucleus> nucleus_;
};

template<class Impl, class... Args>
Generator make_generator(Args&&... args) {
    return Generator(otter::make_otterptr<Impl>(std::forward<Args>(args)...));
}

template <typename T>
static inline T * check_generator(Generator gen) {
    OTTER_CHECK(gen.defined(), "Generator with undefined implementation is not allowed");
//    OTTER_CHECK(T::device() == gen->device(), "Expected a '", T::device(), "' device type for generator but found '", gen->device(), "'");
    return gen.get<T>();
}

template <typename T>
static inline T* get_generator_or_default(const Generator& gen, const Generator& default_gen) {
    return  gen.defined() ? check_generator<T>(gen) : check_generator<T>(default_gen);
}


}   // end namespace otter

#endif /* Generator_hpp */
