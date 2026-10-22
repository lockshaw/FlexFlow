#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_LOSS_FUNCTIONS_LOSS_ATTRS_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_LOSS_FUNCTIONS_LOSS_ATTRS_H

#include "op-attrs/ops/loss_functions/loss_attrs.dtg.h"
#include "utils/record_formatter.h"

namespace FlexFlow {

RecordFormatter loss_attrs_as_dot(LossAttrs const &);

} // namespace FlexFlow

#endif
