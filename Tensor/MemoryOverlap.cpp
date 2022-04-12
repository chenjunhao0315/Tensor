//
//  MemoryOverlap.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/2/17.
//

#include "MemoryOverlap.hpp"
#include "TensorBase.hpp"

namespace otter {

MemOverlap has_internal_overlap(const TensorBase& tensor) {
  return has_internal_overlap(tensor.unsafeGetTensorNucleus());
}

MemOverlap has_internal_overlap(TensorNucleus* t) {
//  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(t->layout() == kStrided);

  if (t->is_non_overlapping_and_dense()) {
    return MemOverlap::NO;
  }

  auto strides = t->strides();
  auto sizes = t->sizes();
  for (const auto i : otter::irange(strides.size())) {
    if (strides[i] == 0 && sizes[i] > 1) {
      return MemOverlap::YES;
    }
  }

  return MemOverlap::TOO_HARD;
}

void assert_no_internal_overlap(const TensorBase& t) {
  assert_no_internal_overlap(t.unsafeGetTensorNucleus());
}

void assert_no_internal_overlap(TensorNucleus* t) {
    (void)t;
    assert(has_internal_overlap(t) != MemOverlap::YES);
    // "unsupported operation: more than one element of the written-to tensor "
    // "refers to a single memory location. Please clone() the tensor before "
    // "performing the operation."
}

MemOverlapStatus get_overlap_status(const TensorBase& a, const TensorBase& b) {
  return get_overlap_status(a.unsafeGetTensorNucleus(), b.unsafeGetTensorNucleus());
}

MemOverlapStatus get_overlap_status(TensorNucleus* a, TensorNucleus* b) {
  if (a == b) return MemOverlapStatus::FULL;
  if (a->numel() == 0 || b->numel() == 0) {
    return MemOverlapStatus::NO;
  }
  if (!a->is_non_overlapping_and_dense() || !b->is_non_overlapping_and_dense()) {
    return MemOverlapStatus::TOO_HARD;
  }
  // Test for memory equality, rather than pointer equality.
  // This reduces precision, but if people are aliasing the
  // same pointer across multiple memorys there are many
  // similar situations (e.g., memory().data() == memory().data()+1)
  // which we will miss.
  auto a_memory = a->unsafe_memory();
  if (a_memory && a_memory.is_alias_of(b->unsafe_memory())) {
    const auto a_begin = static_cast<char*>(a->raw_data());
    const auto a_end = a_begin + a->numel() * a->itemsize();
    const auto b_begin = static_cast<char*>(b->raw_data());
    const auto b_end = b_begin + b->numel() * b->itemsize();

    if (a_begin == b_begin && a_end == b_end) {
      return (a->strides() == b->strides()) ?
          MemOverlapStatus::FULL : MemOverlapStatus::PARTIAL;
    }
    if (a_begin < b_end && b_begin < a_end) {
      return MemOverlapStatus::PARTIAL;
    }
  }
  return MemOverlapStatus::NO;
}

void assert_no_partial_overlap(const TensorBase& a, const TensorBase& b) {
  assert_no_partial_overlap(a.unsafeGetTensorNucleus(), b.unsafeGetTensorNucleus());
}

void assert_no_partial_overlap(TensorNucleus* a, TensorNucleus* b) {
    (void)a;
    (void)b;
    assert(get_overlap_status(a, b) != MemOverlapStatus::PARTIAL);
    // "unsupported operation: some elements of the input tensor and "
    // "the written-to tensor refer to a single memory location. "
    // "Please clone() the tensor before performing the operation."
}

void assert_no_overlap(const TensorBase& a, const TensorBase& b) {
  assert_no_overlap(a.unsafeGetTensorNucleus(), b.unsafeGetTensorNucleus());
}

void assert_no_overlap(TensorNucleus* a, TensorNucleus* b) {
    const auto lap = get_overlap_status(a, b);
    OTTER_CHECK(lap != MemOverlapStatus::PARTIAL && lap != MemOverlapStatus::FULL,
                "unsupported operation: some elements of the input tensor and "
                "the written-to tensor refer to a single memory location. "
                "Please clone() the tensor before performing the operation.");
}

}
