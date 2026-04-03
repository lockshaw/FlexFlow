#include "compiler/machine_mapping/machine_view_2d_projection.h"
#include "utils/containers/filter.h"
#include "utils/containers/scanl.h"
#include "utils/containers/slice.h"
#include "utils/containers/sum.h"
#include "utils/containers/transform.h"
#include "utils/containers/zip3_with_strict.h"
#include "utils/containers/zip_with_strict.h"
#include "utils/nonnegative_int/nonnegative_range.h"
#include "utils/nonnegative_int/num_elements.h"

namespace FlexFlow {

std::vector<MachineSpecificationDimension>
  projection_2d_get_target_dimensions(MachineView2dProjection const &projection)
{
  return transform(projection.dimensions,
                   [](MachineViewDimension const &d) -> MachineSpecificationDimension {
                      return d.target_dim;
                   });
}

std::vector<stride_t>
  projection_2d_get_strides(MachineView2dProjection const &projection)
{
  return transform(projection.dimensions,
                   [](MachineViewDimension const &d) -> stride_t {
                      return d.stride;
                   });
}


MachineSpaceOffset
    projection_2d_get_machine_space_offset(OperatorTaskSpace const &task_space,
                             MachineView2dProjection const &projection,
                             TaskSpaceCoordinate const &coord)
{
  auto get_dimension_indices_for_dimension =
      [&](MachineSpecificationDimension dimension)
      -> std::vector<nonnegative_int> {
    std::vector<MachineSpecificationDimension> mv_dimensions =
        projection_2d_get_target_dimensions(projection);

    return filter(nonnegative_range(num_elements(mv_dimensions)),
                  [&](nonnegative_int idx) {
                    return mv_dimensions.at(idx.unwrap_nonnegative()) ==
                           dimension;
                  });
  };

  auto compute_index =
      [&](std::vector<nonnegative_int> const &dimension_indices) {
        std::vector<stride_t> mv_strides = projection_2d_get_strides(projection);

        std::vector<positive_int> sizes =
            transform(dimension_indices, [&](nonnegative_int i) -> positive_int {
              return task_space.degrees.dims.at(i.unwrap_nonnegative())
                  .positive_int_from_int_ge_two();
            });

        std::vector<nonnegative_int> coord_points =
            transform(dimension_indices, [&](nonnegative_int i) {
              return coord.orthotope_coord.raw.at(i.unwrap_nonnegative());
            });

        std::vector<positive_int> strides =
            transform(dimension_indices, [&](nonnegative_int i) {
              return mv_strides.at(i.unwrap_nonnegative()).unwrapped;
            });

        std::vector<positive_int> scaled_sizes =
          zip_with_strict(sizes, strides,
                          [](positive_int d_i, positive_int s_i) -> positive_int {
                            return d_i * s_i;
                          });

        std::vector<positive_int> coeffs =
            slice(scanl(scaled_sizes, 1_p, std::multiplies<positive_int>()), 0, -1);

        nonnegative_int index_1d = sum(
           zip3_with_strict(coeffs, coord_points, strides,
                     [](positive_int coeff, nonnegative_int coord_point, positive_int stride) -> nonnegative_int {
                        return coeff * coord_point * stride;
                     }));

        return index_1d;
      };

  std::vector<nonnegative_int> inter_dimension_indices =
      get_dimension_indices_for_dimension(
          MachineSpecificationDimension::INTER_NODE);
  std::vector<nonnegative_int> intra_dimension_indices =
      get_dimension_indices_for_dimension(
          MachineSpecificationDimension::INTRA_NODE);

  nonnegative_int node_idx =
      compute_index(inter_dimension_indices);
  nonnegative_int device_idx =
      compute_index(intra_dimension_indices);

  MachineSpaceOffset offset = MachineSpaceOffset{
    /*node_offset=*/node_idx.unwrap_nonnegative(),
    /*device_offset=*/device_idx.unwrap_nonnegative(),
  };

  return offset;
}

} // namespace FlexFlow
