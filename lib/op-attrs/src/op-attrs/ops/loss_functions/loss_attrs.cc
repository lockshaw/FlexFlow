#include "op-attrs/ops/loss_functions/loss_attrs.h"
#include "utils/overload.h"
#include "op-attrs/ops/loss_functions.h"

namespace FlexFlow {

RecordFormatter loss_attrs_as_dot(LossAttrs const &loss_attrs) {
  RecordFormatter result = mk_empty_record(Orientation::VERTICAL);

  result << mk_kv_record("Loss Func", get_loss_function(loss_attrs));

  return result;
}

} // namespace FlexFlow
