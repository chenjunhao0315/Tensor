//
//  RefPtr.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/2/4.
//

#ifndef RefPtr_hpp
#define RefPtr_hpp

#include <utility>

namespace otter {

namespace detail {

template <class TTarget>
struct ptr_quantum_default_null_type final {
    static constexpr TTarget* singleton() noexcept {
        return nullptr;
    }
};

template <class TTarget, class ToNullType, class FromNullType>
TTarget* assign_ptr_(TTarget* rhs) {
    if (FromNullType::singleton() == rhs) {
        return ToNullType::singleton();
    } else {
        return rhs;
    }
}

}

class Ptr_quantum {
    template <typename T, typename NullType>
    friend class Ptr;
    
    template <typename T>
    friend struct ExclusivelyOwnedTraits;
    
protected:
    virtual ~Ptr_quantum() {}
    constexpr Ptr_quantum() noexcept : refCount_(0) {}
    Ptr_quantum(const Ptr_quantum&& other) noexcept : Ptr_quantum() {}
    Ptr_quantum& operator=(const Ptr_quantum&& other) { return *this; }
    Ptr_quantum(const Ptr_quantum& other) noexcept : Ptr_quantum() {}
    Ptr_quantum& operator=(const Ptr_quantum& other) { return *this; }
    
private:
    virtual void release_resources() {}

    mutable int refCount_;
};

struct DontIncreaseRefcount {};

template <class Target, class NullType = detail::ptr_quantum_default_null_type<Target>>
class Ptr {
    template <class TTarget2, class NullType2>
    friend class Ptr;
public:
    Ptr() noexcept : Ptr(NullType::singleton(), DontIncreaseRefcount{}) {}
    
    explicit Ptr(Target *target, DontIncreaseRefcount) noexcept : target_(target) {}
    
    Ptr(const Ptr& other) : target_(other.target_) {
        retain_();
    }
    Ptr(Ptr&& other) : target_(other.target_) {
        other.target_ = NullType::singleton();
    }
    ~Ptr() noexcept {
        reset_();
    }
    
    Ptr& operator=(Ptr&& rhs) & noexcept {
        return operator=<Target, NullType>(std::move(rhs));
    }

    template <class From, class FromNullType>
    Ptr& operator=(Ptr<From, FromNullType>&& rhs) & noexcept {
        Ptr tmp = std::move(rhs);
        this->swap(tmp);
        return *this;
    }

    Ptr& operator=(const Ptr& rhs) & noexcept {
        return operator=<Target, NullType>(rhs);
    }

    template <class From, class FromNullType>
    Ptr& operator=(const Ptr<From, FromNullType>& rhs) & {
        Ptr tmp = rhs;
        this->swap(tmp);
        return *this;
    }
    
    template <class From, class FromNullType>
    Ptr(Ptr<From, FromNullType>&& rhs) noexcept : target_(detail::assign_ptr_<Target, NullType, FromNullType>(rhs.target_)) {
        static_assert(std::is_convertible<From*, Target*>::value);
        rhs.target_ = FromNullType::singleton();
    }
    
    const Target& operator*() const noexcept { return *target_; }
    Target& operator*() noexcept { return *target_; }
    const Target* operator->() const noexcept { return target_; }
    Target* operator->() noexcept { return target_; }
    operator bool() const noexcept { return target_ != NullType::singleton(); }
    
    Target* get() const noexcept {
        return target_;
    }
    
    void reset() {
        reset_();
    }
    
    int use_count() const noexcept {
        return (target_) ? target_->refCount_ : 0;
    }
    
    bool unique() {
        return use_count() == 1;
    }
    
    bool defined() const noexcept {
        return target_ != NullType::singleton();
    }
    
    void swap(Ptr &other) {
        Target* temp = target_;
        target_ = other.target_;
        other.target_ = temp;
    }
    
    // The refCount is not decreased!
    Target* release() noexcept {
        Target* result = target_;
        target_ = NullType::singleton();
        return result;
    }
    
    // The refCount is not increased!
    static Ptr reclaim(Target* ptr) {
        return Ptr(ptr, DontIncreaseRefcount{});
    }
    
    template <class... Args>
    static Ptr make(Args&&... args) {
        return Ptr(new Target(std::forward<Args>(args)...));
    }
    
private:
    void retain_() {
        if (target_ != NullType::singleton())
            ++target_->refCount_;
    }
    void reset_() {
        if (target_ != NullType::singleton() && --target_->refCount_ == 0) {
            target_->release_resources();
            delete target_;
        }
        target_ = NullType::singleton();
    }
    
    explicit Ptr(Target *target) : target_(target) {
        if (target != NullType::singleton()) {
            target_->refCount_ = 1;
        }
    }
    
    Target *target_;
};

template <
    class Target,
    class NullType = detail::ptr_quantum_default_null_type<Target>,
    class... Args>
inline Ptr<Target, NullType> make_otterptr(Args&&... args) {
    return Ptr<Target, NullType>::make(std::forward<Args>(args)...);
}

template <class TTarget1, class NullType1, class TTarget2, class NullType2>
inline bool operator<(
    const Ptr<TTarget1, NullType1>& lhs,
    const Ptr<TTarget2, NullType2>& rhs) noexcept {
  return lhs.get() < rhs.get();
}

template <class TTarget1, class NullType1, class TTarget2, class NullType2>
inline bool operator==(
    const Ptr<TTarget1, NullType1>& lhs,
    const Ptr<TTarget2, NullType2>& rhs) noexcept {
  return lhs.get() == rhs.get();
}

template <class TTarget1, class NullType1, class TTarget2, class NullType2>
inline bool operator!=(
    const Ptr<TTarget1, NullType1>& lhs,
    const Ptr<TTarget2, NullType2>& rhs) noexcept {
  return !operator==(lhs, rhs);
}

}

#endif /* RefPtr_hpp */
