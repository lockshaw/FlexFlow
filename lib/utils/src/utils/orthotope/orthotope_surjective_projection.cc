#include "utils/orthotope/orthotope_surjective_projection.h"
#include "utils/containers/generate_map.h"
#include "utils/containers/get_all_assignments.h"
#include "utils/containers/group_by.h"
#include "utils/containers/map_keys.h"
#include "utils/containers/range.h"
#include "utils/containers/set_of.h"
#include "utils/containers/subvec.h"
#include "utils/containers/sum.h"
#include "utils/containers/transform.h"
#include "utils/containers/unordered_set_of.h"
#include "utils/containers/filter.h"
#include "utils/containers/values.h"
#include "utils/containers/zip_with.h"
#include "utils/exception.h"
#include "utils/orthotope/orthotope.h"
#include "utils/orthotope/orthotope_dim_idx_t.dtg.h"
#include "utils/orthotope/orthotope_dim_idx_t.h"
#include "utils/containers/vector_from_idx_map.h"
#include "utils/containers/scanr.h"
#include "utils/containers/all_of.h"
#include "utils/fmt/vector.h"

namespace FlexFlow {

OrthotopeSurjectiveProjection
  make_orthotope_projection_from_map(std::unordered_map<orthotope_dim_idx_t, orthotope_dim_idx_t> const &m) {
    std::unordered_map<int, orthotope_dim_idx_t> raw_idx_map = map_keys(m, [](orthotope_dim_idx_t const &k) { return k.raw_idx; });
    return OrthotopeSurjectiveProjection{
      /*dim_mapping=*/vector_from_idx_map(raw_idx_map).value(),
      /*reversed=*/false,
    };
}

std::unordered_map<orthotope_dim_idx_t, orthotope_dim_idx_t> get_src_to_dst_dim_map(OrthotopeSurjectiveProjection const &p) {
  if (p.reversed) {
    throw mk_runtime_error(fmt::format("get_src_to_dst_dim_map expected p.reversed=false, but received p={}", p));
  }

  std::unordered_map<int, orthotope_dim_idx_t> raw_idx_map = generate_map(range(p.dim_mapping.size()), [&](int x) { return p.dim_mapping.at(x); });
  return map_keys(raw_idx_map, [](int src_dim_idx) { return orthotope_dim_idx_t{src_dim_idx}; });
}

orthotope_dim_idx_t get_dst_dim_for_src_dim(OrthotopeSurjectiveProjection const &p, orthotope_dim_idx_t const &src_idx) {
  return p.dim_mapping.at(src_idx.raw_idx);
}

int get_src_num_dims(OrthotopeSurjectiveProjection const &p) {
  return p.dim_mapping.size();
}

int get_dst_num_dims(OrthotopeSurjectiveProjection const &p) {
  return unordered_set_of(p.dim_mapping).size();
}

OrthotopeSurjectiveProjection reverse_projection(OrthotopeSurjectiveProjection const &p) {
  OrthotopeSurjectiveProjection result = p;
  result.reversed = !p.reversed;
  return result;
}

std::unordered_set<OrthotopeSurjectiveProjection> get_all_surjective_projections_between(int src_num_dims, int dst_num_dims) {
  if (src_num_dims < dst_num_dims) {
    return transform(get_all_surjective_projections_between(dst_num_dims, src_num_dims), 
                     [](OrthotopeSurjectiveProjection const &p) { return reverse_projection(p); });
  }

  std::set<orthotope_dim_idx_t> src_dim_idxs = dim_idxs_for_orthotope_with_num_dims(src_num_dims);
  std::set<orthotope_dim_idx_t> dst_dim_idxs = dim_idxs_for_orthotope_with_num_dims(dst_num_dims);

  std::unordered_map<orthotope_dim_idx_t, std::unordered_set<orthotope_dim_idx_t>> src_to_dst_idxs = 
    generate_map(src_dim_idxs, [&](orthotope_dim_idx_t) { return unordered_set_of(dst_dim_idxs); });

  std::unordered_set<std::unordered_map<orthotope_dim_idx_t, orthotope_dim_idx_t>> valid_mappings =  
    filter(get_all_assignments(src_to_dst_idxs),
           [&](std::unordered_map<orthotope_dim_idx_t, orthotope_dim_idx_t> const &src_to_dst_idx) { 
             return set_of(values(src_to_dst_idx)) == dst_dim_idxs;
           });

  return transform(valid_mappings, make_orthotope_projection_from_map);
}

int deconflict_overlapping_dims(std::vector<std::pair<int, int>> const &coords_and_sizes) {
  if (coords_and_sizes.size() == 0) {
    throw mk_runtime_error("deconflict_noninjective_dims expected non-empty vector, but receieved empty vector");
  }

  std::vector<int> coords = transform(coords_and_sizes, [](std::pair<int, int> const &p) { return p.first; });
  std::vector<int> dim_sizes = transform(coords_and_sizes, [](std::pair<int, int> const &p) { return p.second; });

  if (!all_of(zip_with(coords, dim_sizes, [](int coord, int dim_size) { return coord > 0 && coord < dim_size; }))) {
    throw mk_runtime_error(fmt::format("coords out of bounds of dim sizes: coords={}, dim_sizes={}", coords, dim_sizes));
  }

  std::vector<int> strides = scanr(subvec(dim_sizes, 1, std::nullopt), 1, [](int next, int accum) { return accum * next; });
  return sum(zip_with(coords, strides, [](int coord, int stride) { return coord * stride; }));
}

OrthotopeCoordinate project_coordinate_through(OrthotopeSurjectiveProjection const &p, Orthotope const &o, OrthotopeCoordinate const &c) {
  if (p.reversed) {
    NOT_IMPLEMENTED(); // TODO @lockshaw
  } else {
    if (c.idxs.size() != get_src_num_dims(p)) {
      throw mk_runtime_error(fmt::format("project_coordinate_through requires projection src and coordinate to have same num dims, but got {} and {} respectively",
                                         get_src_num_dims(p),
                                         c.idxs.size()));
    }

    if (!orthotope_contains_coord(o, c)) {
      throw mk_runtime_error(fmt::format("project_coordinate_through requires coord to be in the orthotope, but got coord={} and orthotope={} respectively", c, o));
    }

    std::unordered_map<orthotope_dim_idx_t, std::set<orthotope_dim_idx_t>> by_dst_dim_idx = 
      group_by(dim_idxs_for_orthotope_with_num_dims(o.dims.size()),
               [&](orthotope_dim_idx_t const &src_dim_idx) { return get_dst_dim_for_src_dim(p, src_dim_idx); });

    
    NOT_IMPLEMENTED();
  }
}

} // namespace FlexFlow
