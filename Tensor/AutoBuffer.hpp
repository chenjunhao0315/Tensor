//
//  AutoBuffer.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/4/30.
//

#ifndef AutoBuffer_hpp
#define AutoBuffer_hpp

#include <iostream>

#include "Exception.hpp"

namespace otter {

template<typename _Tp, size_t fixed_size = 1024 / sizeof(_Tp) + 8>
class AutoBuffer {
public:
    typedef _Tp value_type;
    
    // the default constructor
    AutoBuffer();
    // constructor taking the real buffer size
    explicit AutoBuffer(size_t _size);
    // constructor taking the real buffer size and fill with value
    explicit AutoBuffer(size_t _size, _Tp value);
    
    // the copy constructor
    AutoBuffer(const AutoBuffer<_Tp, fixed_size>& buf);
    // the assignment operator
    AutoBuffer<_Tp, fixed_size>& operator = (const AutoBuffer<_Tp, fixed_size>& buf);
    
    // destructor. calls deallocate()
    ~AutoBuffer();
    
    // allocates the new buffer of size _size. if the _size is small enough, stack-allocated buffer is used
    void allocate(size_t _size);
    // deallocates the buffer if it was dynamically allocated
    void deallocate();
    // resizes the buffer and preserves the content
    void resize(size_t _size);
    // returns the current buffer size
    size_t size() const;
    // returns pointer to the real buffer, stack-allocated or heap-allocated
    inline _Tp* data() { return ptr; }
    // returns read-only pointer to the real buffer, stack-allocated or heap-allocated
    inline const _Tp* data() const { return ptr; }
    
#if !defined(OPENCV_DISABLE_DEPRECATED_COMPATIBILITY) // use to .data() calls instead
    // returns pointer to the real buffer, stack-allocated or heap-allocated
    operator _Tp* () { return ptr; }
    // returns read-only pointer to the real buffer, stack-allocated or heap-allocated
    operator const _Tp* () const { return ptr; }
#else
    // returns a reference to the element at specified location. No bounds checking is performed in Release builds.
    inline _Tp& operator[] (size_t i) {
        OTTER_CHECK(i < sz, "out of range"); return ptr[i];
    }
    // returns a reference to the element at specified location. No bounds checking is performed in Release builds.
    inline const _Tp& operator[] (size_t i) const {
        OTTER_CHECK(i < sz, "out of range"); return ptr[i];
    }
#endif
    
protected:
    // pointer to the real buffer, can point to buf if the buffer is small enough
    _Tp* ptr;
    // size of the real buffer
    size_t sz;
    // pre-allocated buffer. At least 1 element to confirm C++ standard requirements
    _Tp buf[(fixed_size > 0) ? fixed_size : 1];
};

template<typename _Tp, size_t fixed_size>
inline AutoBuffer<_Tp, fixed_size>::AutoBuffer() {
    ptr = buf;
    sz = fixed_size;
}

template<typename _Tp, size_t fixed_size>
inline AutoBuffer<_Tp, fixed_size>::AutoBuffer(size_t _size) {
    ptr = buf;
    sz = fixed_size;
    allocate(_size);
}

template<typename _Tp, size_t fixed_size>
inline AutoBuffer<_Tp, fixed_size>::AutoBuffer(size_t _size, _Tp value) {
    ptr = buf;
    sz = fixed_size;
    allocate(_size);
    for (size_t i = 0; i < _size; ++i) {
        ptr[i] = value;
    }
}

template<typename _Tp, size_t fixed_size>
inline AutoBuffer<_Tp, fixed_size>::AutoBuffer(const AutoBuffer<_Tp, fixed_size>& abuf ) {
    ptr = buf;
    sz = fixed_size;
    allocate(abuf.size());
    for( size_t i = 0; i < sz; i++ )
        ptr[i] = abuf.ptr[i];
}

template<typename _Tp, size_t fixed_size>
inline AutoBuffer<_Tp, fixed_size>& AutoBuffer<_Tp, fixed_size>::operator = (const AutoBuffer<_Tp, fixed_size>& abuf) {
    if( this != &abuf )
    {
        deallocate();
        allocate(abuf.size());
        for( size_t i = 0; i < sz; i++ )
            ptr[i] = abuf.ptr[i];
    }
    return *this;
}

template<typename _Tp, size_t fixed_size>
inline AutoBuffer<_Tp, fixed_size>::~AutoBuffer() {
    deallocate();
}

template<typename _Tp, size_t fixed_size>
inline void AutoBuffer<_Tp, fixed_size>::allocate(size_t _size) {
    if(_size <= sz) {
        sz = _size;
        return;
    }
    deallocate();
    sz = _size;
    if(_size > fixed_size) {
        ptr = new _Tp[_size];
    }
}

template<typename _Tp, size_t fixed_size>
inline void AutoBuffer<_Tp, fixed_size>::deallocate() {
    if( ptr != buf ) {
        delete[] ptr;
        ptr = buf;
        sz = fixed_size;
    }
}

template<typename _Tp, size_t fixed_size> inline void
AutoBuffer<_Tp, fixed_size>::resize(size_t _size) {
    if(_size <= sz) {
        sz = _size;
        return;
    }
    size_t i, prevsize = sz, minsize = std::min(prevsize, _size);
    _Tp* prevptr = ptr;
    
    ptr = _size > fixed_size ? new _Tp[_size] : buf;
    sz = _size;
    
    if( ptr != prevptr )
        for( i = 0; i < minsize; i++ )
            ptr[i] = prevptr[i];
    for( i = prevsize; i < _size; i++ )
        ptr[i] = _Tp();
    
    if( prevptr != buf )
        delete[] prevptr;
}

template<typename _Tp, size_t fixed_size> inline size_t
AutoBuffer<_Tp, fixed_size>::size() const {
    return sz;
}

}   // end namespace otter

#endif /* AutoBuffer_hpp */
