//
//  MemoryOverlap.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/2/17.
//

#ifndef MemoryOverlap_hpp
#define MemoryOverlap_hpp

namespace otter {

class TensorBase;
struct TensorNucleus;

enum class MemOverlap { NO, YES, TOO_HARD };

enum class MemOverlapStatus { FULL, PARTIAL, NO, TOO_HARD };

MemOverlap has_internal_overlap(const TensorBase& t);
MemOverlap has_internal_overlap(TensorNucleus* t);

void assert_no_internal_overlap(const TensorBase& t);
void assert_no_internal_overlap(TensorNucleus* t);

MemOverlapStatus get_overlap_status(const TensorBase& a, const TensorBase& b);
MemOverlapStatus get_overlap_status(TensorNucleus* a, TensorNucleus* b);

void assert_no_partial_overlap(const TensorBase& a, const TensorBase& b);
void assert_no_partial_overlap(TensorNucleus* a, TensorNucleus* b);

void assert_no_overlap(const TensorBase& a, const TensorBase& b);
void assert_no_overlap(TensorNucleus* a, TensorNucleus* b);

}

#endif /* MemoryOverlap_hpp */
