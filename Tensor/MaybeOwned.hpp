//
//  MaybeOwned.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/1/29.
//

#ifndef MaybeOwned_hpp
#define MaybeOwned_hpp

#include <cstddef>
#include <type_traits>
#include <utility>

#include "in_place.hpp"

namespace otter {

template <typename T>
struct MaybeOwnedTraitsNucleus {
    using owned_type = T;
    using borrow_type = const T *;
    
    static borrow_type create_borrow(const owned_type& from) {
        return &from;
    }
    
    static void assign_borrow(borrow_type& lhs, borrow_type rhs) {
        lhs = rhs;
    }
    
    static void destory_borrow(borrow_type& /*toDestroy*/) {}
    
    static const owned_type& referenceFromBorrow(const borrow_type& borrow) {
        return *borrow;
    }
    
    static const owned_type* pointerFromBorrow(const borrow_type& borrow) {
        return borrow;
    }
};

template <typename T>
struct MaybeOwnedTraits;


template <typename T>
class MaybeOwned {
private:
    using owned_type = typename MaybeOwnedTraits<T>::owned_type;
    using borrow_type = typename MaybeOwnedTraits<T>::borrow_type;
    
    bool isBorrowed_;
    union {
        owned_type own_;
        borrow_type borrow_;
    };
    
    explicit MaybeOwned(const owned_type& t)
    : isBorrowed_(true), borrow_(MaybeOwnedTraits<T>::create_borrow(t)) {}
    
    explicit MaybeOwned(T&& t) noexcept(std::is_nothrow_move_constructible<T>::value) : isBorrowed_(false), own_(std::move(t)) {}
    
    template <class... Args>
    explicit MaybeOwned(in_place_t, Args&&... args) : isBorrowed_(false), own_(std::forward<Args>(args)...) {}
    
public:
    MaybeOwned() : isBorrowed_(true), borrow_() {}
    
    MaybeOwned(const MaybeOwned& rhs) : isBorrowed_(rhs.isBorrowed_) {
        if (rhs.isBorrowed_) {
            MaybeOwnedTraits<T>::assign_borrow(borrow_, rhs.borrow_);
        } else {
            ::new (&own_) T(rhs.own_);
        }
    }
    
    MaybeOwned& operator=(const MaybeOwned& rhs) {
        if (this == &rhs) {
            return *this;
        }
        if (!isBorrowed_) {
            if (rhs.isBorrowed_) {
                own_.~T();
                MaybeOwnedTraits<T>::assign_borrow(borrow_, rhs.borrow_);
                isBorrowed_ = true;
            } else {
                own_ = rhs.own_;
            }
        } else {
            if (rhs.isBorrowed_) {
                MaybeOwnedTraits<T>::assign_borrow(borrow_, rhs.borrow_);
            } else {
                MaybeOwnedTraits<T>::destroy_borrow(borrow_);
                new (&own_) T(rhs.own_);
                isBorrowed_ = false;
            }
        }
        //        TORCH_INTERNAL_ASSERT_DEBUG_ONLY(isBorrowed_ == rhs.isBorrowed_);
        return *this;
    }
    
    MaybeOwned(MaybeOwned&& rhs) noexcept(std::is_nothrow_move_constructible<T>::value) : isBorrowed_(rhs.isBorrowed_) {
        if (rhs.isBorrowed_) {
            MaybeOwnedTraits<T>::assign_borrow(borrow_, rhs.borrow_);
        } else {
            new (&own_) T(std::move(rhs.own_));
        }
    }
    
    MaybeOwned& operator=(MaybeOwned&& rhs) noexcept(std::is_nothrow_move_assignable<T>::value) {
        if (this == &rhs) {
            return *this;
        }
        if (!isBorrowed_) {
            if (rhs.isBorrowed_) {
                own_.~T();
                MaybeOwnedTraits<T>::assign_borrow(borrow_, rhs.borrow_);
                isBorrowed_ = true;
            } else {
                own_ = std::move(rhs.own_);
            }
        } else {
            if (rhs.isBorrowed_) {
                MaybeOwnedTraits<T>::assign_borrow(borrow_, rhs.borrow_);
            } else {
                MaybeOwnedTraits<T>::destroy_borrow(borrow_);
                new (&own_) T(std::move(rhs.own_));
                isBorrowed_ = false;
            }
        }
        //        TORCH_INTERNAL_ASSERT_DEBUG_ONLY(isBorrowed_ == rhs.isBorrowed_);
        return *this;
    }
    
    static MaybeOwned borrowed(const T& t) {
        return MaybeOwned(t);
    }
    
    static MaybeOwned owned(T&& t) noexcept(std::is_nothrow_move_constructible<T>::value) {
        return MaybeOwned(std::move(t));
    }
    
    template <class... Args>
    static MaybeOwned owned(in_place_t, Args&&... args) {
        return MaybeOwned(in_place, std::forward<Args>(args)...);
    }
    
    ~MaybeOwned() {
        if (!isBorrowed_) {
            own_.~T();
        } else {
            MaybeOwnedTraits<T>::destroy_borrow(borrow_);
        }
    }
    
    bool unsafeIsBorrowed() const {
        return isBorrowed_;
    }
    
    const T& operator*() const& {
        return (isBorrowed_) ? MaybeOwnedTraits<T>::referenceFromBorrow(borrow_) : own_;
    }
        
    const T* operator->() const {
        return (isBorrowed_) ? MaybeOwnedTraits<T>::pointerFromBorrow(borrow_) : &own_;
    }
        
    T operator*() && {
        if (isBorrowed_) {
            return MaybeOwnedTraits<T>::referenceFromBorrow(borrow_);
        } else {
            return std::move(own_);
        }
    }
};
    
}   // end namespace otter

#endif /* MaybeOwned_hpp */
