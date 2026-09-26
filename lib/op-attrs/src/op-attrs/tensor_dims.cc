#include "op-attrs/tensor_dims.h"
#include "op-attrs/ff_ordered/ff_ordered_enumerate.h"
#include "op-attrs/ff_ordered/ff_ordered_filtrans.h"
#include "op-attrs/ff_ordered/ff_ordered_get_idxs.h"
#include "op-attrs/ff_ordered/ff_ordered_slice.h"
#include "op-attrs/ff_ordered/ff_ordered_zip.h"
#include "op-attrs/ff_ordered/ff_ordered_zip_with.h"
#include "op-attrs/replica_parallel_dim_set.h"
#include "op-attrs/shard_parallel_dim.dtg.h"
#include "utils/containers/all_are_true.h"
#include "utils/containers/all_of.h"
#include "utils/containers/cartesian_product.h"
#include "utils/containers/contains.h"
#include "utils/containers/product.h"
#include "utils/containers/reversed.h"
#include "utils/containers/set_of.h"
#include "utils/containers/transform.h"
#include "utils/containers/vector_of.h"
#include "utils/containers/zip.h"
#include "utils/integer_conversions.h"
#include "utils/nonnegative_int/nonnegative_range.h"
#include "utils/nonnegative_int/num_elements.h"
#include "utils/containers/are_all_same.h"
#include "utils/containers/minimum.h"
#include "utils/containers/get_only.h"
#include "utils/containers/take_while.h"
#include "op-attrs/ff_ordered/ff_ordered_concat.h"
#include "op-attrs/relative_ff_dim_t.h"
#include "op-attrs/ff_ordered/ff_ordered_transform_with_idx.h"
#include "op-attrs/ff_ordered/ff_ordered_filtrans_with_idx.h"

namespace FlexFlow {

FFOrdered<positive_int> const &ff_ordered(TensorDims const &dims) {
  return dims.ff_ordered;
}

bool tensor_dims_has_dim(TensorDims const &tensor_dims, ff_dim_t dim) {
  return contains(ff_ordered_get_idxs(tensor_dims.ff_ordered), dim);
}

num_tensor_dims_t get_num_dims(TensorDims const &dims) {
  return num_tensor_dims_t{
      num_elements(dims.ff_ordered),
  };
}

positive_int dim_at_idx(TensorDims const &dims, relative_ff_dim_t idx) {
  return dims.ff_ordered.at(idx);
}

positive_int &dim_at_idx(TensorDims &dims, relative_ff_dim_t idx) {
  return dims.ff_ordered.at(idx);
}

positive_int dim_at_idx(TensorDims const &dims, ff_dim_t ff_dim_idx) {
  return dims.ff_ordered.at(ff_dim_idx);
}

positive_int &dim_at_idx(TensorDims &dims, ff_dim_t ff_dim_idx) {
  return dims.ff_ordered.at(ff_dim_idx);
}

std::optional<positive_int> try_dim_at_idx(TensorDims const &dims,
                                           relative_ff_dim_t idx) {
  if (dims.ff_ordered.idx_is_valid(idx)) {
    return dims.ff_ordered.at(idx);
  } else {
    return std::nullopt;
  }
}

std::optional<positive_int> try_dim_at_idx(TensorDims const &dims,
                                           ff_dim_t idx) {
  if (dims.ff_ordered.idx_is_valid(idx)) {
    return dims.ff_ordered.at(idx);
  } else {
    return std::nullopt;
  }
}

positive_int get_num_elements(TensorDims const &d) {
  return product(d.ff_ordered);
}

bool tensor_dims_is_broadcastable_to(TensorDims const &curr,
                                     TensorDims const &goal) {
  if (get_num_dims(curr) > get_num_dims(goal)) {
    return false;
  }

  std::vector<positive_int> curr_dims = vector_of(curr.ff_ordered);
  std::vector<positive_int> goal_dims = vector_of(goal.ff_ordered);

  for (auto const &[curr_dim, goal_dim] :
       zip(reversed(curr_dims), reversed(goal_dims))) {
    if (curr_dim != 1 && curr_dim != goal_dim) {
      return false;
    }
  }

  return true;
}

bool tensor_dims_contains_coord(TensorDims const &tensor_dims,
                                TensorDimsCoord const &coord) {
  ASSERT(coord.ff_ordered.size() == get_num_dims(tensor_dims));

  return all_are_true(ff_ordered_zip_with(
      coord.ff_ordered,
      tensor_dims.ff_ordered,
      [](nonnegative_int const &coord_entry, positive_int const &dim_size) {
        return coord_entry < dim_size;
      }));
}

TensorDimsCoord get_broadcast_src_coord(TensorDims const &input_dims,
                                        TensorDims const &output_dims,
                                        TensorDimsCoord const &dst_coord) {
  ASSERT(tensor_dims_contains_coord(output_dims, dst_coord),
         output_dims,
         dst_coord);
  ASSERT(tensor_dims_is_broadcastable_to(input_dims, output_dims),
         input_dims,
         output_dims);

  relative_ff_dim_t trailing_start_idx = relative_ff_dim_t{
      -1 * get_num_dims(input_dims).int_from_num_tensor_dims()};

  FFOrdered<nonnegative_int> trailing_entries =
      ff_ordered_slice(dst_coord.ff_ordered, trailing_start_idx);

  FFOrdered<positive_int> trailing_dims =
      ff_ordered_slice(output_dims.ff_ordered, trailing_start_idx);

  TensorDimsCoord result = TensorDimsCoord{
      ff_ordered_zip_with(trailing_entries,
                          input_dims.ff_ordered,
                          [](nonnegative_int const &coord_entry,
                             positive_int const &input_dim_size) {
                            if (input_dim_size == 1) {
                              return 0_n;
                            } else {
                              return coord_entry;
                            }
                          }),
  };

  ASSERT(tensor_dims_contains_coord(input_dims, result),
         output_dims,
         dst_coord,
         input_dims,
         result);

  return result;
}

TensorDims
  get_shared_leading_dims(std::set<TensorDims> const &not_reduced)
{
  num_tensor_dims_t min_dims = minimum(
    transform(not_reduced,
              [&](TensorDims const &d) -> num_tensor_dims_t {
                return get_num_dims(d);
              }));

  nonnegative_int
    num_leading_dims = num_elements(
      take_while(
        ff_dim_range(min_dims.nonnegative_int_from_num_tensor_dims()),
        [&](ff_dim_t d) -> bool {
          std::set<positive_int> dim_sizes =
            transform(not_reduced,
                      [&](TensorDims const &t) -> positive_int {
                        return dim_at_idx(t, d);
                      });

          return are_all_same(dim_sizes);
        }));

  ff_dim_t first_nonleading_dim = ff_dim_t{num_leading_dims};

  return get_only(
    transform(
      not_reduced,
      [&](TensorDims const &d) -> TensorDims {
        return slice_tensor_dims(d, ff_dim_t{0_n}, first_nonleading_dim);        
      }));
}

TensorDims
  tensor_dims_remove_leading_dims(TensorDims const &d,
                                  TensorDims const &leading_dims)
{
  nonnegative_int num_leading_dims = 
    get_num_dims(leading_dims).nonnegative_int_from_num_tensor_dims();

  ff_dim_t first_nonleading_dim = ff_dim_t{num_leading_dims};

  TensorDims d_leading_dims = slice_tensor_dims(d, ff_dim_t{0_n}, first_nonleading_dim);
  ASSERT(leading_dims == d_leading_dims);

  return slice_tensor_dims(d, first_nonleading_dim, std::nullopt);
}

TensorDims
  tensor_dims_remove_trailing_dims(TensorDims const &d,
                                   TensorDims const &trailing_dims)
{
  num_tensor_dims_t d_num_dims = get_num_dims(d);

  nonnegative_int num_trailing_dims = 
    get_num_dims(trailing_dims).nonnegative_int_from_num_tensor_dims();

  ff_dim_t first_trailing_dim = 
    ff_dim_t_from_relative_ff_dim_t(
      relative_ff_dim_t{-1 * num_trailing_dims.int_from_nonnegative_int()},
      d_num_dims);

  TensorDims d_trailing_dims = slice_tensor_dims(d, first_trailing_dim, std::nullopt);
  ASSERT(trailing_dims == d_trailing_dims);

  return slice_tensor_dims(d, ff_dim_t{0_n}, first_trailing_dim);
}

std::set<TensorDimsCoord>
    get_tensor_dims_coord_set(TensorDims const &tensor_dims) {
  std::vector<std::vector<nonnegative_int>> per_dim_ranges = transform(
      vector_of(tensor_dims.ff_ordered),
      [](positive_int dim_size) -> std::vector<nonnegative_int> {
        return nonnegative_range(dim_size.nonnegative_int_from_positive_int());
      });

  std::set<std::vector<nonnegative_int>> raw_points =
      set_of(cartesian_product(per_dim_ranges));

  return transform(raw_points,
                   [](std::vector<nonnegative_int> const &raw_point) {
                     return TensorDimsCoord{ff_ordered_of(raw_point)};
                   });
}

std::set<ff_dim_t> get_ff_dim_t_set(TensorDims const &tensor_dims) {
  return set_of(ff_ordered_get_idxs(tensor_dims.ff_ordered));
}

std::optional<TensorDims>
    get_broadcast_target_dims(std::set<TensorDims> const &dims) {
  for (TensorDims target_candidate : dims) {
    if (all_of(dims, [&](TensorDims const &d) {
          return tensor_dims_is_broadcastable_to(d, target_candidate);
        })) {
      return target_candidate;
    }
  }

  return std::nullopt;
}

TensorDims tensor_dims_drop_dims(
    TensorDims const &dims,
    std::function<bool(ff_dim_t)> const &should_drop_dim) {
  std::vector<positive_int> result;
  for (ff_dim_t idx : ff_ordered_get_idxs(dims.ff_ordered)) {
    if (!should_drop_dim(idx)) {
      result.push_back(dims.ff_ordered.at(idx));
    }
  }

  return TensorDims{ff_ordered_of(result)};
}

TensorDims concat_tensor_dims(TensorDims const &leading,
                              TensorDims const &trailing)
{
  return TensorDims{
    ff_ordered_concat(leading.ff_ordered, trailing.ff_ordered),
  };
}

TensorDims slice_tensor_dims(TensorDims const &dims,
                             relative_ff_dim_t const &start,
                             std::optional<relative_ff_dim_t> const &stop) {
  return TensorDims{
      ff_ordered_slice(dims.ff_ordered, start, stop),
  };
}

TensorDims slice_tensor_dims(TensorDims const &dims,
                             ff_dim_t const &start,
                             std::optional<ff_dim_t> const &stop) {
  return TensorDims{
      ff_ordered_slice(dims.ff_ordered, start, stop),
  };
}

TensorDims tensor_dims_transform_with_idx(
  TensorDims const &dims,
  std::function<positive_int(ff_dim_t, positive_int)> const &f)
{
  return TensorDims{
    ff_ordered_transform_with_idx(dims.ff_ordered, f),
  };
}

TensorDims tensor_dims_filtrans_with_idx(
  TensorDims const &dims,
  std::function<std::optional<positive_int>(ff_dim_t, positive_int)> const &f)
{
  return TensorDims{
    ff_ordered_filtrans_with_idx(dims.ff_ordered, f),
  };
}

} // namespace FlexFlow
