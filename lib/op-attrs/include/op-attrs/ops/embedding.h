#ifndef _FLEXFLOW_EMBEDDING_ATTRS_H
#define _FLEXFLOW_EMBEDDING_ATTRS_H

#include "op-attrs/initializer_attrs.dtg.h"
#include "op-attrs/ops/core.h"
#include "op-attrs/ops/embedding_attrs.dtg.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "op-attrs/tensor_shape.h"
#include "utils/record_formatter.h"
#include <tl/expected.hpp>

namespace FlexFlow {

CHECK_VALID_OP_ATTR(EmbeddingAttrs);

RecordFormatter as_dot(EmbeddingAttrs const &);

tl::expected<TensorShape, std::string> get_output_shape(EmbeddingAttrs const &,
                                                        TensorShape const &);
tl::expected<TensorShape, std::string> get_weights_shape(EmbeddingAttrs const &,
                                                         TensorShape const &);

tl::expected<ParallelTensorShape, std::string>
    get_output_shape(EmbeddingAttrs const &, ParallelTensorShape const &);
tl::expected<ParallelTensorShape, std::string>
    get_weights_shape(EmbeddingAttrs const &, ParallelTensorShape const &);

/**
 * @brief Chosen to match pytorch
 *
 * see
 * https://github.com/pytorch/pytorch/blob/1eba9b3aa3c43f86f4a2c807ac8e12c4a7767340/torch/nn/modules/sparse.py#L180-L182
 */
std::vector<InitializerAttrs> get_initializers(
    EmbeddingAttrs const &,
    std::optional<InitializerAttrs> const &initializer_attrs = std::nullopt);

} // namespace FlexFlow

#endif
