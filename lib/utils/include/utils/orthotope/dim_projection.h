#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_ORTHOTOPE_DIM_PROJECTION_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_ORTHOTOPE_DIM_PROJECTION_H

#include "utils/orthotope/down_projection.dtg.h"
#include "utils/orthotope/eq_projection.dtg.h"
#include "utils/orthotope/up_projection.dtg.h"

namespace FlexFlow {

template <typename T1, typename T2, typename T3>
EqProjection<T1, T3> compose_dim_projections(EqProjection<T1, T2> const &fst, EqProjection<T2, T3> const &snd) {
  return EqProjection{
    exhaustive_relational_join(fst.dim_mapping, snd.dim_mapping)
  };
}

template <typename T1, typename T2, typename T3>
UpProjection<T1, T3> compose_dim_projections(UpProjection<T1, T2> const &fst, UpProjection<T2, T3> const &snd) {
  NOT_IMPLEMENTED();
}

template <typename T1, typename T2, typename T3>
DownProjection<T1, T3> compose_dim_projections(DownProjection<T1, T2> const &fst, DownProjection<T2, T3> const &snd) {
  NOT_IMPLEMENTED();
}

template <typename T1, typename T2, typename T3>
UpProjection<T1, T3> compose_dim_projections(EqProjection<T1, T2> const &fst, UpProjection<T2, T3> const &snd) {
  NOT_IMPLEMENTED();
}

template <typename T1, typename T2, typename T3>
UpProjection<T1, T3> compose_dim_projections(UpProjection<T1, T2> const &fst, EqProjection<T2, T3> const &snd) {
  NOT_IMPLEMENTED();
}

template <typename T1, typename T2, typename T3>
DownProjection<T1, T3> compose_dim_projections(EqProjection<T1, T2> const &fst, DownProjection<T2, T3> const &snd) {
  NOT_IMPLEMENTED();
}

template <typename T1, typename T2, typename T3>
DownProjection<T1, T3> compose_dim_projections(DownProjection<T1, T2> const &fst, EqProjection<T2, T3> const &snd) {
  NOT_IMPLEMENTED();
}

} // namespace FlexFlow

#endif
