//
//  TensorShape.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/2/10.
//

#ifndef TensorShape_hpp
#define TensorShape_hpp

#include "ArrayRef.hpp"

namespace otter {

class Tensor;

namespace native {

Tensor select(const Tensor& self, int64_t dim, int64_t index);

Tensor permute(const Tensor& self, IntArrayRef dims);

Tensor& transpose_(Tensor& self, int64_t dim0, int64_t dim1);
Tensor transpose(const Tensor& self, int64_t dim0, int64_t dim1);

Tensor expand(const Tensor& self, IntArrayRef sizes);
Tensor expand_as(const Tensor& self, const Tensor& other);

Tensor alias(const Tensor& self);

Tensor repeat(const Tensor& self, IntArrayRef repeats);

Tensor stack(TensorList tensors, int64_t dim);

Tensor view(const Tensor& self, IntArrayRef sizes);

Tensor reshape(const Tensor& self, IntArrayRef sizes);
Tensor reshape_as(const Tensor& self, const Tensor& other);

Tensor detach(const Tensor& self);

Tensor as_strided_tensorimpl(const Tensor& self, IntArrayRef size, IntArrayRef stride);
Tensor as_strided_tensorimpl(const Tensor& self, IntArrayRef size, IntArrayRef stride, int64_t memory_offset_);
const Tensor &as_strided_(const Tensor& self, IntArrayRef size, IntArrayRef stride);
const Tensor &as_strided_(const Tensor& self, IntArrayRef size, IntArrayRef stride, int64_t memory_offset_);

Tensor slice(const Tensor& self, int64_t dim, int64_t start = INT64_MAX, int64_t end = 0, int64_t step = 1);

Tensor unsqueeze(const Tensor& self, int64_t dim);
Tensor& unsqueeze_(Tensor& self, int64_t dim);

Tensor squeeze(const Tensor& self, int64_t dim);
Tensor& squeeze_(Tensor& self, int64_t dim);

Tensor flatten(const Tensor& self, int64_t start_dim = 0, int64_t end_dim = -1);

Tensor narrow(const Tensor& self, int64_t dim, int64_t start, int64_t length);

Tensor& cat_out(TensorList tensors, int64_t dim, Tensor& out);
Tensor cat(TensorList tensors, int64_t dim);

Tensor diagonal(const Tensor& self, int64_t offset, int64_t dim1_, int64_t dim2_);

Tensor unfold(const Tensor& self, int64_t dim, int64_t size, int64_t step);

std::vector<Tensor> chunk(const Tensor& self, int64_t chunks, int64_t dim = 0);

std::vector<Tensor> tensor_split(const Tensor& self, int64_t sections, int64_t dim);
std::vector<Tensor> tensor_split(const Tensor& self, IntArrayRef indices, int64_t dim);
std::vector<Tensor> tensor_split(const Tensor& self, const Tensor& tensor_indices_or_sections, int64_t dim);

std::vector<Tensor> split(const Tensor& self, int64_t split_size, int64_t dim);
std::vector<Tensor> split(const Tensor& self, IntArrayRef sizes, int64_t dim);

std::vector<Tensor> hsplit(const Tensor& self, int64_t split_size);
std::vector<Tensor> vsplit(const Tensor& self, int64_t split_size);
std::vector<Tensor> dsplit(const Tensor& self, int64_t split_size);

std::vector<Tensor> split_with_sizes(const Tensor& self, IntArrayRef split_sizes, int64_t dim);

std::vector<Tensor> hsplit(const Tensor& self, IntArrayRef split_sizes);
std::vector<Tensor> vsplit(const Tensor& self, IntArrayRef split_sizes);
std::vector<Tensor> dsplit(const Tensor& self, IntArrayRef split_sizes);

inline int64_t get_num_splits(const Tensor& self, int64_t split_size, int64_t dim) {
    OTTER_CHECK(self.dim() != 0, "split expects at least a 1-dimensional tensor");
    OTTER_CHECK(split_size >= 0,  "split expects split_size be non-negative, but got split_size=", split_size);
    int64_t dim_size = self.size(dim);
    OTTER_CHECK(split_size > 0 || dim_size == 0,
                "split_size can only be 0 if dimension size is 0, "
                "but got dimension size of ", dim_size);
    // if split_size is 0 and dimension size is 0, there is 1 split.
    int64_t num_splits = 1;
    if (split_size != 0) {
        // ensuring num_splits is at least 1 makes consistent the case where split_size > dim_size
        // (returns a single split).  We might want to error here, but keep it for BC.
        num_splits = std::max<int64_t>((dim_size + split_size - 1) / split_size, 1);
    }
    return num_splits;
}

}   // namespace native

}   // namespace otter

#endif /* TensorShape_hpp */
