//
//  ExclusivelyOwned.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/1/31.
//

#ifndef ExclusivelyOwned_hpp
#define ExclusivelyOwned_hpp

#include <utility>

#include "in_place.hpp"

namespace otter {

template <typename T>
struct ExclusivelyOwnedTraits;

template <typename T>
class ExclusivelyOwned {
    using EOT = ExclusivelyOwnedTraits<T>;
    union {
        char dummy_;
        typename ExclusivelyOwnedTraits<T>::repr_type repr_;
    };
    
public:
    ExclusivelyOwned() : repr_(EOT::nullRepr()) {}
    
    explicit ExclusivelyOwned(T&& t) : repr_(EOT::moveToRepr(std::move(t))) {}
    
    template <class... Args>
    explicit ExclusivelyOwned(in_place_t, Args&&... args)
    : repr_(EOT::createInPlace(std::forward<Args>(args)...)) {}
    
    ExclusivelyOwned(const ExclusivelyOwned&) = delete;
    
    ExclusivelyOwned(ExclusivelyOwned&& rhs) noexcept
    : repr_(std::move(rhs.repr_)) {
        rhs.repr_ = EOT::nullRepr();
    }
    
    ExclusivelyOwned& operator=(const ExclusivelyOwned&) = delete;
    
    ExclusivelyOwned& operator=(ExclusivelyOwned&& rhs) noexcept {
        EOT::destroyOwned(repr_);
        repr_ = std::move(rhs.repr_);
        rhs.repr_ = EOT::nullRepr();
        return *this;
    }
    
    ExclusivelyOwned& operator=(T&& rhs) noexcept {
        EOT::destroyOwned(repr_);
        repr_ = EOT::moveToRepr(std::move(rhs));
        return *this;
    }
    
    ~ExclusivelyOwned() {
        EOT::destroyOwned(repr_);
    }
    
    // We don't provide this because it would require us to be able to
    // differentiate an owned-but-empty T from a lack of T. This is
    // particularly problematic for Tensor, which wants to use an
    // undefined Tensor as its null state.
    explicit operator bool() const noexcept = delete;
    
    operator T() && {
        return take();
    }
    
    T take() && {
        return EOT::take(repr_);
    }
    
    typename EOT::const_pointer_type operator->() const {
        return get();
    }
    
    typename EOT::const_pointer_type get() const {
        return EOT::getImpl(repr_);
    }
    
    typename EOT::pointer_type operator->() {
        return get();
    }
    
    typename EOT::pointer_type get() {
        return EOT::getImpl(repr_);
    }
    
    std::remove_pointer_t<typename EOT::const_pointer_type>& operator*() const {
        return *get();
    }
    
    std::remove_pointer_t<typename EOT::pointer_type>& operator*() {
        return *get();
    }
};


}

#endif /* ExclusivelyOwned_hpp */
