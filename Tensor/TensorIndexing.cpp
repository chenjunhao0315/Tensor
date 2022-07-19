//
//  TensorIndexing.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/7/16.
//

#include "TensorIndexing.hpp"
#include "Tensor.hpp"

namespace otter {

namespace indexing {
const EllipsisIndexType Ellipsis = EllipsisIndexType();
std::ostream& operator<<(std::ostream& stream, const Slice& slice) {
    stream << slice.start() << ":" << slice.stop() << ":" << slice.step();
    return stream;
}
std::ostream& operator<<(std::ostream& stream, const TensorIndex& tensor_index) {
    if (tensor_index.is_none()) {
        stream << "None";
    } else if (tensor_index.is_ellipsis()) {
        stream << "...";
    } else if (tensor_index.is_integer()) {
        stream << tensor_index.integer();
    } else if (tensor_index.is_boolean()) {
        stream << std::boolalpha << tensor_index.boolean();
    } else if (tensor_index.is_slice()) {
        stream << tensor_index.slice();
    } else if (tensor_index.is_tensor()) {
        stream << tensor_index.tensor();
    }
    return stream;
}
std::ostream& operator<<(std::ostream& stream, const std::vector<TensorIndex>& tensor_indices) {
    stream << "(";
    for (const auto i : otter::irange(tensor_indices.size())) {
        stream << tensor_indices[i];
        if (i < tensor_indices.size() - 1) stream << ", ";
    }
    stream << ")";
    return stream;
}
// This mirrors `THPVariable_setitem` in torch/csrc/autograd/python_variable_indexing.cpp
// for "the assigned value is a Scalar" case
static inline void set_item(const Tensor& self, ArrayRef<TensorIndex> indices, const Scalar& v) {
    Tensor value;
    {
        Device self_device = self.device();
        // TODO: This qint special case looks very suspicious...
        value = otter::indexing::scalarToTensor(v, self.options(), self_device);
    }
    return set_item(self, indices, value);
}
} // namespace indexing

Tensor Tensor::index(ArrayRef<otter::indexing::TensorIndex> indices) const {
    OTTER_CHECK(indices.size() > 0, "Passing an empty index list to Tensor::index() is not valid syntax");
    return otter::indexing::get_item(*this, indices);
}
Tensor Tensor::index(std::initializer_list<otter::indexing::TensorIndex> indices) const {
    return index(ArrayRef<otter::indexing::TensorIndex>(indices));
}
Tensor & Tensor::index_put_(ArrayRef<otter::indexing::TensorIndex> indices, Tensor const & rhs) {
    OTTER_CHECK(indices.size() > 0, "Passing an empty index list to Tensor::index_put_() is not valid syntax");
    otter::indexing::set_item(*this, indices, rhs);
    return *this;
}
Tensor & Tensor::index_put_(ArrayRef<otter::indexing::TensorIndex> indices, const Scalar& v) {
    OTTER_CHECK(indices.size() > 0, "Passing an empty index list to Tensor::index_put_() is not valid syntax");
    otter::indexing::set_item(*this, indices, v);
    return *this;
}
Tensor & Tensor::index_put_(std::initializer_list<otter::indexing::TensorIndex> indices, Tensor const & rhs) {
    return index_put_(ArrayRef<otter::indexing::TensorIndex>(indices), rhs);
}
Tensor & Tensor::index_put_(std::initializer_list<otter::indexing::TensorIndex> indices, const Scalar& v) {
    return index_put_(ArrayRef<otter::indexing::TensorIndex>(indices), v);
}

}   // end namespace otter
