#include "utils/orthotope/bounded_component.h"

namespace FlexFlow {

BoundedComponent trivial_bounded_component() {
  return BoundedComponent{
    /*component=*/0_n,
    /*bound=*/1_p,
  };
}

} // namespace FlexFlow
