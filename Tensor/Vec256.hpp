//
//  Vec256.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/2/17.
//

#ifndef Vec256_h
#define Vec256_h

#include "VecIntrinsic.hpp"
#include "VecBase.hpp"
#include "Vec256_float.hpp"

namespace otter {
namespace vec{

template <typename T>
std::ostream& operator<<(std::ostream& stream, const Vectorized<T>& vec) {
    T buf[Vectorized<T>::size()];
    vec.store(buf);
    stream << "vec[";
    for (int i = 0; i != Vectorized<T>::size(); i++) {
        if (i != 0) {
            stream << ", ";
        }
        stream << buf[i];
    }
    stream << "]";
    return stream;
}

}   // end namespace vec
}   // end namespace otter


#endif /* Vec256_h */
