#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_ORTHOTOPE_MINIMAL_DIM_DOMAIN_MAPPING_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_ORTHOTOPE_MINIMAL_DIM_DOMAIN_MAPPING_H

#include "utils/bidict/algorithms/exhaustive_relational_join.h"
#include "utils/bidict/algorithms/left_entries.h"
#include "utils/bidict/algorithms/right_entries.h"
#include "utils/bidict/algorithms/transform_keys.h"
#include "utils/bidict/algorithms/transform_values.h"
#include "utils/bidict/bidict.h"
#include "utils/bidict/generate_bidict.h"
#include "utils/hash/tuple.h"
#include "utils/orthotope/dim_coord.dtg.h"
#include "utils/orthotope/dim_coord.h"
#include "utils/orthotope/dim_domain_hemiunique_mapping.h"
#include "utils/orthotope/dim_domain_mapping.h"
#include "utils/orthotope/dim_ordering.dtg.h"
#include "utils/orthotope/dim_projection.h"
#include "utils/orthotope/minimal_dim_domain.dtg.h"
#include "utils/orthotope/minimal_dim_domain_mapping.h"
#include "utils/relation/compose_hemiunique_binary_relations.h"
#include "utils/relation/hemiunique_binary_relation.h"
#include "utils/relation/hemiunique_binrel_transform_l_and_r.h"

namespace FlexFlow {

template <typename L, typename R>
struct MinimalDimDomainHemiuniqueMapping {
public:
  explicit MinimalDimDomainHemiuniqueMapping(
      HemiuniqueBinaryRelation<DimCoord<L>, DimCoord<R>> const &coord_mapping,
      MinimalDimDomain<L> const &l_domain,
      MinimalDimDomain<R> const &r_domain)
      : coord_mapping(coord_mapping), l_domain(l_domain), r_domain(r_domain) {

    ASSERT(get_coords_in_minimal_dim_domain(l_domain) ==
           coord_mapping.left_entries());

    ASSERT(get_coords_in_minimal_dim_domain(r_domain) ==
           coord_mapping.right_entries());
  }

  bool operator==(MinimalDimDomainHemiuniqueMapping<L, R> const &other) const {
    return this->tie() == other.tie();
  }

  bool operator!=(MinimalDimDomainHemiuniqueMapping<L, R> const &other) const {
    return this->tie() != other.tie();
  }

public:
  HemiuniqueBinaryRelation<DimCoord<L>, DimCoord<R>> coord_mapping;
  MinimalDimDomain<L> l_domain;
  MinimalDimDomain<R> r_domain;

private:
  std::tuple<decltype(coord_mapping) const &,
             decltype(l_domain) const &,
             decltype(r_domain) const &>
      tie() const {
    return std::tie(this->coord_mapping, this->l_domain, this->r_domain);
  }

  friend struct ::std::hash<MinimalDimDomainHemiuniqueMapping<L, R>>;
};

template <typename L, typename R>
std::string format_as(MinimalDimDomainHemiuniqueMapping<L, R> const &m) {
  CHECK_FMTABLE(L);
  CHECK_FMTABLE(R);

  return fmt::format("<MinimalDimDomainHemiuniqueMapping l_domain={} "
                     "r_domain={} coord_mapping={}>",
                     m.l_domain,
                     m.r_domain,
                     m.coord_mapping);
}

template <typename L, typename R>
std::ostream &operator<<(std::ostream &s,
                         MinimalDimDomainHemiuniqueMapping<L, R> const &m) {
  CHECK_FMTABLE(L);
  CHECK_FMTABLE(R);

  return (s << fmt::to_string(m));
}

template <typename L, typename R>
MinimalDimDomainHemiuniqueMapping<L, R>
    minimal_hemiunique_mapping_from_dim_domain_mapping(
        DimDomainHemiuniqueMapping<L, R> const &m) {

  std::unordered_set<L> l_nontrivial_dims =
      get_nontrivial_domain_dims(m.l_domain);

  std::unordered_set<R> r_nontrivial_dims =
      get_nontrivial_domain_dims(m.r_domain);

  return MinimalDimDomainHemiuniqueMapping{
      /*coord_mapping=*/
      hemiunique_binrel_transform_l_and_r(
          m.coord_mapping,
          [&](DimCoord<L> const &l_coord) {
            return restrict_coord_to_dims(l_coord, l_nontrivial_dims);
          },
          [&](DimCoord<R> const &r_coord) {
            return restrict_coord_to_dims(r_coord, r_nontrivial_dims);
          }),
      /*l_domain=*/minimal_dim_domain_from_dim_domain(m.l_domain),
      /*r_domain=*/minimal_dim_domain_from_dim_domain(m.r_domain),
  };
}

template <typename L, typename R>
DimDomainHemiuniqueMapping<L, R>
    dim_domain_hemiunique_mapping_from_minimal_dim_domain(
        MinimalDimDomainHemiuniqueMapping<L, R> const &m,
        std::unordered_set<L> const &l_trivial_dims,
        std::unordered_set<R> const &r_trivial_dims) {

  DimDomain<L> l_domain =
      dim_domain_from_minimal_dim_domain(m.l_domain, l_trivial_dims);
  DimDomain<R> r_domain =
      dim_domain_from_minimal_dim_domain(m.r_domain, r_trivial_dims);

  std::unordered_set<L> all_l_dims = get_domain_dims(l_domain);
  std::unordered_set<R> all_r_dims = get_domain_dims(r_domain);

  return DimDomainHemiuniqueMapping{
      /*coord_mapping=*/
      hemiunique_binrel_transform_l_and_r(
          m.coord_mapping,
          [&](DimCoord<L> const &l_coord) {
            return lift_dim_coord(l_coord, all_l_dims);
          },
          [&](DimCoord<R> const &r_coord) {
            return lift_dim_coord(r_coord, all_r_dims);
          }),
      /*l_domain=*/l_domain,
      /*r_domain=*/r_domain,
  };
}

template <typename L, typename R>
MinimalDimDomainHemiuniqueMapping<L, R>
    empty_minimal_dim_domain_hemiunique_mapping() {
  return MinimalDimDomainHemiuniqueMapping{
      /*coord_mapping=*/{},
      /*l_domain=*/empty_minimal_dim_domain<L>(),
      /*r_domain=*/empty_minimal_dim_domain<R>(),
  };
}

template <typename L, typename R>
MinimalDimDomainHemiuniqueMapping<L, R>
    minimal_dim_domain_hemiunique_mapping_identity_map(
        MinimalDimDomain<L> const &l_domain,
        MinimalDimDomain<R> const &r_domain,
        DimOrdering<L> const &l_dim_ordering,
        DimOrdering<R> const &r_dim_ordering) {
  DimProjection<L, R> projection =
      dim_projection_identity_map(lift_minimal_dim_domain(l_domain),
                                  lift_minimal_dim_domain(r_domain),
                                  l_dim_ordering,
                                  r_dim_ordering);

  return minimal_dim_domain_hemiunique_mapping_from_projection(
      /*projection=*/projection,
      /*l_domain=*/l_domain,
      /*r_domain=*/r_domain,
      /*l_dim_ordering=*/l_dim_ordering,
      /*r_dim_ordering=*/r_dim_ordering);
}

template <typename L, typename R>
MinimalDimDomainHemiuniqueMapping<R, L>
    invert_minimal_dim_domain_hemiunique_mapping(
        MinimalDimDomainHemiuniqueMapping<L, R> const
            &minimal_dim_domain_mapping) {

  return MinimalDimDomainHemiuniqueMapping{
      /*coord_mapping=*/minimal_dim_domain_mapping.coord_mapping.reversed(),
      /*l_domain=*/minimal_dim_domain_mapping.r_domain,
      /*r_domain=*/minimal_dim_domain_mapping.l_domain,
  };
}

template <typename T1, typename T2, typename T3>
MinimalDimDomainHemiuniqueMapping<T1, T3>
    compose_minimal_dim_domain_hemiunique_mappings(
        MinimalDimDomainHemiuniqueMapping<T1, T2> const &lhs,
        MinimalDimDomainHemiuniqueMapping<T2, T3> const &rhs) {

  ASSERT(lhs.r_domain == rhs.l_domain);

  return MinimalDimDomainHemiuniqueMapping{
      /*coord_mapping=*/compose_hemiunique_binary_relations(lhs.coord_mapping,
                                                            rhs.coord_mapping),
      /*l_domain=*/lhs.l_domain,
      /*r_domain=*/rhs.r_domain,
  };
}

template <typename T1, typename T2, typename T3>
DimDomainHemiuniqueMapping<T1, T3>
    compose_dim_domain_hemiunique_mappings_through_minimal(
        DimDomainHemiuniqueMapping<T1, T2> const &lhs,
        DimDomainHemiuniqueMapping<T2, T3> const &rhs) {

  MinimalDimDomainHemiuniqueMapping<T1, T2> minimal_lhs =
      minimal_hemiunique_mapping_from_dim_domain_mapping(lhs);

  std::unordered_set<T1> t1_trivial_dims =
      get_trivial_domain_dims(lhs.l_domain);

  MinimalDimDomainHemiuniqueMapping<T2, T3> minimal_rhs =
      minimal_hemiunique_mapping_from_dim_domain_mapping(rhs);

  std::unordered_set<T3> t3_trivial_dims =
      get_trivial_domain_dims(rhs.r_domain);

  return dim_domain_hemiunique_mapping_from_minimal_dim_domain(
      compose_minimal_dim_domain_hemiunique_mappings(minimal_lhs, minimal_rhs),
      t1_trivial_dims,
      t3_trivial_dims);
}

template <typename L, typename R>
MinimalDimDomainHemiuniqueMapping<L, R>
    minimal_dim_domain_hemiunique_mapping_from_projection(
        DimProjection<L, R> const &projection,
        MinimalDimDomain<L> const &l_domain,
        MinimalDimDomain<R> const &r_domain,
        DimOrdering<L> const &l_dim_ordering,
        DimOrdering<R> const &r_dim_ordering) {

  return MinimalDimDomainHemiuniqueMapping{
      /*coord_mapping=*/
      HemiuniqueBinaryRelation<DimCoord<L>, DimCoord<R>>{
          generate_bidict(
              get_coords_in_minimal_dim_domain(l_domain),
              [&](DimCoord<L> const &l_coord) {
                return compute_dim_projection(
                    /*projection=*/projection,
                    /*input_coord=*/l_coord,
                    /*input_domain=*/lift_minimal_dim_domain(l_domain),
                    /*output_domain=*/lift_minimal_dim_domain(r_domain),
                    /*input_dim_ordering=*/l_dim_ordering,
                    /*output_dim_ordering=*/r_dim_ordering);
              }),
      },
      /*l_domain=*/l_domain,
      /*r_domain=*/r_domain,
  };
}

} // namespace FlexFlow

namespace std {

template <typename L, typename R>
struct hash<::FlexFlow::MinimalDimDomainHemiuniqueMapping<L, R>> {
  size_t operator()(::FlexFlow::MinimalDimDomainHemiuniqueMapping<L, R> const
                        &minimal_dim_domain_mapping) const {
    return get_std_hash(minimal_dim_domain_mapping.tie());
  }
};

} // namespace std

#endif
