#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_ORTHOTOPE_DIM_PROJECTION_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_ORTHOTOPE_DIM_PROJECTION_H

#include "utils/orthotope/dim_projection.dtg.h"
#include "utils/orthotope/down_projection.h"
#include "utils/orthotope/eq_projection.h"
#include "utils/orthotope/up_projection.h"
#include "utils/orthotope/dim_coord.dtg.h"
#include "utils/overload.h"

namespace FlexFlow {

template <typename L, typename R> 
std::unordered_set<L> input_dims_of_projection(DimProjection<L, R> const &projection) {
  return projection.template visit<std::unordered_set<L>>(overload {
    [](UpProjection<L, R> const &p) { return input_dims_of_up_projection(p); },
    [](EqProjection<L, R> const &p) { return input_dims_of_eq_projection(p); },
    [](DownProjection<L, R> const &p) { return input_dims_of_down_projection(p); },
  }); 
}

template <typename L, typename R>
std::unordered_set<R> output_dims_of_projection(DimProjection<L, R> const &projection) {
  return projection.template visit<std::unordered_set<R>>(overload {
    [](UpProjection<L, R> const &p) { return output_dims_of_up_projection(p); },
    [](EqProjection<L, R> const &p) { return output_dims_of_eq_projection(p); },
    [](DownProjection<L, R> const &p) { return output_dims_of_down_projection(p); },
  }); 
};

// template <typename L, typename R>
// DimCoord<R> compute_projection(DimProjection<L, R> const &projection, DimCoord<L> const &coord) {
//   if (coord_dims(coord) != 
// }

} // namespace FlexFlow

#endif
