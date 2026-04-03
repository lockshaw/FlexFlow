#include "compiler/machine_mapping/machine_view_1d_projection.h"
#include "utils/containers/scanl.h"
#include "utils/containers/scanr.h"
#include "utils/containers/slice.h"
#include "utils/containers/transform.h"
#include "utils/containers/zip3_with_strict.h"
#include "utils/containers/sum.h"
#include "utils/containers/zip_with_strict.h"

namespace FlexFlow {

std::vector<stride_t> projection_1d_get_strides(MachineView1dProjection const &projection) {
  return projection.strides;
}

FlatMachineSpaceOffset
    projection_1d_get_flat_machine_space_offset(OperatorTaskSpace const &task_space,
                             MachineView1dProjection const &projection,
                             TaskSpaceCoordinate const &coord)
{
  std::vector<positive_int> sizes =
    transform(task_space.degrees.dims,
              [](int_ge_two s) -> positive_int {
                return s.positive_int_from_int_ge_two();
              });

  std::vector<nonnegative_int> coord_points = coord.orthotope_coord.raw;

  std::vector<positive_int> strides = transform(
    projection.strides,
    [](stride_t s) -> positive_int {
      return s.unwrapped;
    });

  std::vector<positive_int> scaled_sizes =
    zip_with_strict(sizes, strides,
                    [](positive_int d_i, positive_int s_i) -> positive_int {
                      return d_i * s_i;
                    });

  std::vector<positive_int> coeffs =
      slice(scanr(scaled_sizes, 1_p, std::multiplies<positive_int>()), 1, std::nullopt);

  nonnegative_int index_1d = sum(
     zip3_with_strict(coeffs, coord_points, strides,
               [](positive_int coeff, nonnegative_int coord_point, positive_int stride) -> nonnegative_int {
                  return coeff * coord_point * stride;
               }));

  return FlatMachineSpaceOffset{index_1d.unwrap_nonnegative()};
}

} // namespace FlexFlow
