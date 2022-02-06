//
//  DType.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/1/29.
//

#ifndef DType_hpp
#define DType_hpp

#include "ScalarType.hpp"

namespace otter {

class _Uninitialized {};

struct TypeMetaData final {
    using New = void*();
    using PlacementNew = void(void*, size_t);
    using Copy = void(const void*, void*, size_t);
    using PlacementDelete = void(void*, size_t);
    using Delete = void(void*);
    
    constexpr TypeMetaData() noexcept
          : itemsize_(0),
            new_(nullptr),
            placementNew_(nullptr),
            copy_(nullptr),
            placementDelete_(nullptr),
            delete_(nullptr) {}
    
    constexpr TypeMetaData(
          size_t itemsize,
          New* newFn,
          PlacementNew* placementNew,
          Copy* copy,
          PlacementDelete* placementDelete,
          Delete* deleteFn) noexcept
          : itemsize_(itemsize),
            new_(newFn),
            placementNew_(placementNew),
            copy_(copy),
            placementDelete_(placementDelete),
            delete_(deleteFn) {}
    
    size_t itemsize_;
    New* new_;
    PlacementNew* placementNew_;
    Copy* copy_;
    PlacementDelete* placementDelete_;
    Delete* delete_;
};

template <typename T>
inline void* _New() {
    return new T;
}

template <typename T, std::enable_if_t<std::is_default_constructible<T>::value>* = nullptr>
inline constexpr TypeMetaData::New* _PickNew() {
    return &_New<T>;
}

template <typename T>
inline void _PlacementNew(void* ptr, size_t n) {
    T* typed_ptr = static_cast<T*>(ptr);
    for (size_t i = 0; i < n; ++i) {
        new (typed_ptr + i) T;
    }
}

template <typename T, std::enable_if_t<std::is_default_constructible<T>::value>* = nullptr>
inline constexpr TypeMetaData::PlacementNew* _PickPlacementNew() {
    return (std::is_fundamental<T>::value || std::is_pointer<T>::value)
        ? nullptr
        : &_PlacementNew<T>;
}

template <typename T>
inline void _Copy(const void* src, void* dst, size_t n) {
    const T* typed_src = static_cast<const T*>(src);
    T* typed_dst = static_cast<T*>(dst);
    for (size_t i = 0; i < n; ++i) {
        typed_dst[i] = typed_src[i];
    }
}

template <typename T, std::enable_if_t<std::is_copy_assignable<T>::value>* = nullptr>
inline constexpr TypeMetaData::Copy* _PickCopy() {
    return (std::is_fundamental<T>::value || std::is_pointer<T>::value)
        ? nullptr
        : &_Copy<T>;
}

template <typename T>
inline void _PlacementDelete(void* ptr, size_t n) {
    T* typed_ptr = static_cast<T*>(ptr);
    for (size_t i = 0; i < n; ++i) {
        typed_ptr[i].~T();
    }
}

template <typename T>
inline constexpr TypeMetaData::PlacementDelete* _PickPlacementDelete() {
    return (std::is_fundamental<T>::value || std::is_pointer<T>::value)
        ? nullptr
        : &_PlacementDelete<T>;
}

template <typename T>
inline void _Delete(void* ptr) {
    T* typed_ptr = static_cast<T*>(ptr);
    delete typed_ptr;
}

template <class T>
inline constexpr TypeMetaData::Delete* _PickDelete() noexcept {
    return &_Delete<T>;
}

struct TypeMeta {
public:
    using New = TypeMetaData::New;
    using PlacementNew = TypeMetaData::PlacementNew;
    using Copy = TypeMetaData::Copy;
    using PlacementDelete = TypeMetaData::PlacementDelete;
    using Delete = TypeMetaData::Delete;
    
//    TypeMeta() {}
    
    TypeMeta() noexcept;
    TypeMeta(const TypeMeta& src) noexcept = default;
    TypeMeta& operator=(const TypeMeta& src) noexcept = default;
    TypeMeta(TypeMeta&& rhs) noexcept = default;
    inline TypeMeta& operator=(ScalarType scalar_type) noexcept {
        index_ = static_cast<uint16_t>(scalar_type);
        return *this;
    }
    
    friend bool operator==(const TypeMeta lhs, const TypeMeta rhs) noexcept;
    
    #define MaxTypeIndex 32
    static TypeMetaData* typeMetaDatas();
    
    template <typename T>
    static TypeMeta Make() {
        return TypeMeta(type2index<T>());
    }
    
    template <typename T>
    bool Match() const noexcept {
        return (*this == Make<T>());
    }
    
    inline size_t itemsize() const noexcept {
        return data().itemsize_;
    }
    
    New* newFn() const noexcept {
        return data().new_;
    }
    
    PlacementNew* placementNew() const noexcept {
        return data().placementNew_;
    }
    
    Copy* copy() const noexcept {
        return data().copy_;
    }
    
    PlacementDelete* placementDelete() const noexcept {
        return data().placementDelete_;
    }
    
    Delete* deleteFn() const noexcept {
        return data().delete_;
    }
    
    template <class T>
    static constexpr size_t ItemSize() noexcept {
        return sizeof(T);
    }
    
    static inline TypeMeta fromScalarType(ScalarType scalar_type) {
        const auto index = static_cast<uint16_t>(scalar_type);
        return TypeMeta(index);
    }

    inline ScalarType toScalarType() const {
        return static_cast<ScalarType>(index_);
    }
private:
    explicit TypeMeta(const uint16_t index) noexcept : index_(index) {}
    
    template <class T>
    static uint16_t type2index() noexcept;
    
    inline const TypeMetaData& data() const {
        return typeMetaDatas()[index_];
    }
    
    uint16_t index_;
};

#define DEFINE_SCALAR_METADATA_INSTANCE(T, name)                \
    template <>                                                 \
    constexpr uint16_t TypeMeta::type2index<T>() noexcept {     \
        return static_cast<uint16_t>(ScalarType::name);         \
    }
    OTTER_ALL_SCALAR_TYPES(DEFINE_SCALAR_METADATA_INSTANCE)
#undef DEFINE_SCALAR_METADATA_INSTANCE

template <>
constexpr uint16_t TypeMeta::type2index<_Uninitialized>() noexcept {
    return static_cast<uint16_t>(ScalarType::Undefined);
}

inline TypeMeta::TypeMeta() noexcept : index_(type2index<_Uninitialized>()) {}

static inline TypeMeta scalarTypeToTypeMeta(ScalarType scalar_type) {
  return TypeMeta::fromScalarType(scalar_type);
}

static inline ScalarType typeMetaToScalarType(TypeMeta dtype) {
  return dtype.toScalarType();
}

inline bool operator==(const TypeMeta lhs, const TypeMeta rhs) noexcept {
    return (lhs.index_ == rhs.index_);
}

inline bool operator!=(const TypeMeta lhs, const TypeMeta rhs) noexcept {
    return !operator==(lhs, rhs);
}

}   // end namespace otter

#endif /* DType_hpp */
