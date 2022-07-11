//
//  TensorAdvancedIndexing.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/7/11.
//

#include "TensorAdvancedIndexing.hpp"

namespace otter {

//template <bool use_new_options = false, typename Meta>
//void scatter_meta_impl(
//    Meta& meta,
//    const Tensor& self,
//    int64_t dim,
//    const Tensor& index,
//    const c10::optional<Tensor>& src = nullopt,
//    const c10::optional<c10::string_view> reduce = nullopt) {
//  int64_t wrapped_dim = at::maybe_wrap_dim(dim, self.dim());
//  at::native::scatter_gather_dtype_check("scatter", self, index, src);
//  at::native::scatter_shape_check(self, wrapped_dim, index, src);
//  auto output = meta.maybe_get_output(0);
//  if (output.defined()) {
//    at::assert_no_internal_overlap(output);
//    at::assert_no_overlap(output, index);
//    if (src.has_value()) {
//      at::assert_no_overlap(output, src.value());
//    }
//  }
//  meta.set_output_raw_strided(0, self.sizes(), {}, self.options());
//  if (reduce.has_value()) {
//    // Check if we have a valid reduce operator.
//    get_operator_enum(reduce.value(), use_new_options);
//  }
//}
//TORCH_META_FUNC2(scatter, src)
//(const Tensor& self, int64_t dim, const Tensor& index, const Tensor& src) {
//  scatter_meta_impl(*this, self, dim, index, src);
//}
//TORCH_META_FUNC2(scatter, value)
//(const Tensor& self, int64_t dim, const Tensor& index, const Scalar& value) {
//  scatter_meta_impl(*this, self, dim, index);
//}
//TORCH_META_FUNC2(scatter, reduce)
//(const Tensor& self,
// int64_t dim,
// const Tensor& index,
// const Tensor& src,
// const int64_t reduce) {
//  scatter_meta_impl(*this, self, dim, index, src, reduce);
//}
//TORCH_META_FUNC2(scatter, value_reduce)
//(const Tensor& self,
// int64_t dim,
// const Tensor& index,
// const Scalar& src,
// const int64_t reduce) {
//  scatter_meta_impl(*this, self, dim, index, nullopt, reduce);
//}
//
//DEFINE_DISPATCH(scatter_stub);
//DEFINE_DISPATCH(scatter_fill_stub);
//DEFINE_DISPATCH(scatter_reduce_stub);
//DEFINE_DISPATCH(scatter_scalar_reduce_stub);
//
//template <bool use_new_options = false, typename T, typename ReduceStub, typename FillStub>
//void scatter_impl(
//    const Tensor& self,
//    int64_t dim,
//    const Tensor& index,
//    const T& src,
//    const Tensor& out,
//    ReduceStub& reduce_stub,
//    FillStub& fill_stub,
//    const c10::optional<c10::string_view> reduce = nullopt,
//    bool reduce_includes_self = true) {
//  dim = at::maybe_wrap_dim(dim, self.dim());
//  auto mut_out = const_cast<Tensor&>(out);
//  if (!self.is_same(mut_out)) {
//    mut_out.copy_(self);
//  }
//  if (index.numel() == 0) return;
//  if (reduce.has_value()) {
//    auto op = meta::get_operator_enum(reduce.value(), use_new_options);
//    if (!reduce_includes_self) {
//      // scatter inits for reduction to appropriate indices (used by scatter_reduce.two)
//      scatter_reduce_exclude_self_helper(mut_out, dim, index, op);
//    }
//    reduce_stub(self.device().type(), mut_out, dim, index, src, op);
//  } else {
//    fill_stub(self.device().type(), mut_out, dim, index, src);
//  }
//}
//TORCH_IMPL_FUNC(scatter_src_out)
//(const Tensor& self,
// int64_t dim,
// const Tensor& index,
// const Tensor& src,
// const Tensor& out) {
//  scatter_impl(self, dim, index, src, out,
//               scatter_reduce_stub,
//               scatter_stub);
//}
//TORCH_IMPL_FUNC(scatter_value_out)
//(const Tensor& self,
// int64_t dim,
// const Tensor& index,
// const Scalar& value,
// const Tensor& out) {
//  scatter_impl(self, dim, index, value, out,
//               scatter_scalar_reduce_stub,
//               scatter_fill_stub);
//}
//TORCH_IMPL_FUNC(scatter_reduce_out)
//(const Tensor& self,
// int64_t dim,
// const Tensor& index,
// const Tensor& src,
// const int64_t reduce,
// const Tensor& out) {
//  scatter_impl(self, dim, index, src, out,
//               scatter_reduce_stub,
//               scatter_stub,
//               reduce);
//}
//TORCH_IMPL_FUNC(scatter_value_reduce_out)
//(const Tensor& self,
// int64_t dim,
// const Tensor& index,
// const Scalar& value,
// const int64_t reduce,
// const Tensor& out) {
//  scatter_impl(self, dim, index, value, out,
//               scatter_scalar_reduce_stub,
//               scatter_fill_stub,
//               reduce);

}   // end namespace otter
