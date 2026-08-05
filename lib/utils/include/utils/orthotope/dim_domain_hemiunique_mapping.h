#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_ORTHOTOPE_DIM_DOMAIN_HEMIUNIQUE_MAPPING_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_ORTHOTOPE_DIM_DOMAIN_HEMIUNIQUE_MAPPING_H

#include "utils/bidict/algorithms/exhaustive_relational_join.h"
#include "utils/bidict/algorithms/filter_bidict.h"
#include "utils/bidict/algorithms/left_entries.h"
#include "utils/bidict/algorithms/right_entries.h"
#include "utils/bidict/algorithms/unstructured_relation_from_bidict.h"
#include "utils/bidict/bidict.h"
#include "utils/bidict/generate_bidict.h"
#include "utils/containers/get_only.h"
#include "utils/hash/tuple.h"
#include "utils/orthotope/dim_coord.dtg.h"
#include "utils/orthotope/dim_coord.h"
#include "utils/orthotope/dim_domain.dtg.h"
#include "utils/orthotope/dim_domain_biunique_mapping.h"
#include "utils/orthotope/dim_domain_hemiunique_mapping.h"
#include "utils/orthotope/dim_ordering.dtg.h"
#include "utils/orthotope/dim_projection.h"
#include "utils/orthotope/minimal_dim_domain.dtg.h"
#include "utils/relation/hemiunique_binary_relation.h"
#include "utils/relation/uniqueness.dtg.h"

namespace FlexFlow {

/**
 * \brief A left-total and right-total binary relation between a pair of
 *        \ref DimDomain ""s.
 */
template <typename L, typename R>
struct DimDomainHemiuniqueMapping {
public:
  explicit DimDomainHemiuniqueMapping(
      HemiuniqueBinaryRelation<DimCoord<L>, DimCoord<R>> const &coord_mapping,
      DimDomain<L> const &l_domain,
      DimDomain<R> const &r_domain)
      : coord_mapping(coord_mapping), l_domain(l_domain), r_domain(r_domain) {

    // check that the mapping is left-total
    ASSERT(get_coords_in_dim_domain(l_domain) == coord_mapping.left_entries());

    // check that the mapping is right-total
    ASSERT(get_coords_in_dim_domain(r_domain) == coord_mapping.right_entries());
  }

  bool operator==(DimDomainHemiuniqueMapping<L, R> const &other) const {
    return this->tie() == other.tie();
  }

  bool operator!=(DimDomainHemiuniqueMapping<L, R> const &other) const {
    return this->tie() != other.tie();
  }

  Uniqueness get_uniqueness() const {
    return this->coord_mapping.get_uniqueness();
  }

  bidict<DimCoord<L>, DimCoord<R>> const &require_biunique() const {
    return this->coord_mapping.require_biunique();
  }

  OneToMany<L, R> require_strictly_left_unique() const {
    return this->coord_mapping.require_strictly_left_unique();
  }

  OneToMany<L, R> require_strictly_right_unique() const {
    return this->coord_mapping.require_strictly_right_unique();
  }

public:
  HemiuniqueBinaryRelation<DimCoord<L>, DimCoord<R>> coord_mapping;
  DimDomain<L> l_domain;
  DimDomain<R> r_domain;

private:
  std::tuple<decltype(coord_mapping) const &,
             decltype(l_domain) const &,
             decltype(r_domain) const &>
      tie() const {
    return std::tie(this->coord_mapping, this->l_domain, this->r_domain);
  }

  friend struct ::std::hash<DimDomainHemiuniqueMapping<L, R>>;
};

template <typename L, typename R>
std::string format_as(DimDomainHemiuniqueMapping<L, R> const &m) {
  CHECK_FMTABLE(L);
  CHECK_FMTABLE(R);

  return fmt::format(
      "<DimDomainHemiuniqueMapping l_domain={} r_domain={} coord_mapping={}>",
      m.l_domain,
      m.r_domain,
      m.coord_mapping);
}

template <typename L, typename R>
std::ostream &operator<<(std::ostream &s,
                         DimDomainHemiuniqueMapping<L, R> const &m) {
  CHECK_FMTABLE(L);
  CHECK_FMTABLE(R);

  return (s << fmt::to_string(m));
}

/**
 * \brief Create an \ref DimDomainHemiuniqueMapping between a pair of empty \ref DimDomain ""s.
 *
 * \relates DimDomainHemiuniqueMapping
 */
template <typename L, typename R>
DimDomainHemiuniqueMapping<L, R> empty_dim_domain_hemiunique_mapping() {
  return DimDomainHemiuniqueMapping<L, R>{
      /*coord_mapping=*/HemiuniqueBinaryRelation{
          bidict<DimCoord<L>, DimCoord<R>>{
              {
                  DimCoord<L>{{}},
                  DimCoord<R>{{}},
              },
          },
      },
      /*l_domain=*/empty_dim_domain<L>(),
      /*r_domain=*/empty_dim_domain<R>(),
  };
}

template <typename L, typename R>
DimDomainHemiuniqueMapping<L, R> hemiunique_from_biunique_dim_domain_mapping(
    DimDomainBiuniqueMapping<L, R> const &m) {
  return DimDomainHemiuniqueMapping<L, R>{
      /*coord_mapping=*/HemiuniqueBinaryRelation<DimCoord<L>, DimCoord<R>>{
          m.coord_mapping},
      /*l_domain=*/m.l_domain,
      /*r_domain=*/m.r_domain,
  };
}

/**
 * \brief Create a \ref DimDomainHemiuniqueMapping between two equi-dimensional \ref DimDomain ""s,
 *        that is simply a permutation of the dimension labels (i.e., the closest thing
 *        possible to an identity relation).
 *
 * \relates DimDomainHemiuniqueMapping
 */
template <typename L, typename R>
DimDomainHemiuniqueMapping<L, R> dim_domain_hemiunique_mapping_identity_map(
    DimDomain<L> const &l_domain,
    DimDomain<R> const &r_domain,
    DimOrdering<L> const &l_dim_ordering,
    DimOrdering<R> const &r_dim_ordering) {
  DimProjection<L, R> projection = dim_projection_identity_map(
      l_domain, r_domain, l_dim_ordering, r_dim_ordering);

  return dim_domain_hemiunique_mapping_from_projection(
      /*projection=*/projection,
      /*l_domain=*/l_domain,
      /*r_domain=*/r_domain,
      /*l_dim_ordering=*/l_dim_ordering,
      /*r_dim_ordering=*/r_dim_ordering);
}

/**
 * \brief Invert a \ref DimDomainHemiuniqueMapping.
 *
 * \relates DimDomainHemiuniqueMapping
 */
template <typename L, typename R>
DimDomainHemiuniqueMapping<R, L> invert_dim_domain_hemiunique_mapping(
    DimDomainHemiuniqueMapping<L, R> const &dim_domain_mapping) {

  return DimDomainHemiuniqueMapping{
      /*coord_mapping=*/dim_domain_mapping.coord_mapping.inverted(),
      /*l_domain=*/dim_domain_mapping.r_domain,
      /*r_domain=*/dim_domain_mapping.l_domain,
  };
}

/**
 * \brief Compose two \ref DimDomainHemiuniqueMapping ""s.
 *
 * \note The mappings must have their internal dimension in common, a la matrix multiplication.
 *
 * \relates DimDomainHemiuniqueMapping
 */
template <typename T1, typename T2, typename T3>
DimDomainHemiuniqueMapping<T1, T3> compose_dim_domain_hemiunique_mappings(
    DimDomainHemiuniqueMapping<T1, T2> const &lhs,
    DimDomainHemiuniqueMapping<T2, T3> const &rhs) {
  // TODO(@lockshaw)(#pr):
  NOT_IMPLEMENTED();
}

template <typename L, typename R>
DimDomainHemiuniqueMapping<L, R>
    dim_domain_hemiunique_mapping_by_scaling_projection(
        DimProjection<L, R> const &projection,
        DimDomain<L> const &l_domain,
        DimDomain<R> const &r_domain,
        DimOrdering<L> const &l_dim_ordering,
        DimOrdering<R> const &r_dim_ordering) {
  EqProjection<L, R> eq_proj = projection.require_eq_proj();

  auto have_differing_dim_sizes = [&](L l_dim, R r_dim) -> bool {
    positive_int l_size = l_domain.dims.at(l_dim);
    positive_int r_size = r_domain.dims.at(r_dim);

    return l_size != r_size;
  };

  bidict<L, R> with_matching_dim_sizes =
      filter_bidict(eq_proj.dim_mapping, [&](L const &l, R const &r) -> bool {
        return !have_differing_dim_sizes(l, r);
      });

  bidict<L, R> with_differing_dim_sizes =
      filter_bidict(eq_proj.dim_mapping, have_differing_dim_sizes);

  ASSERT(with_differing_dim_sizes.size() == 1);

  std::pair<L, R> differing_dim_pair =
      get_only(unstructured_relation_from_bidict(with_differing_dim_sizes));

  L differing_l_dim = differing_dim_pair.first;
  R differing_r_dim = differing_dim_pair.second;

  positive_int differing_l_dim_size = l_domain.dims.at(differing_l_dim);
  positive_int differing_r_dim_size = r_domain.dims.at(differing_r_dim);

  if (differing_r_dim_size > differing_l_dim_size) {
    return invert_dim_domain_hemiunique_mapping(
        dim_domain_hemiunique_mapping_by_scaling_projection(
            invert_dim_projection(projection),
            /*l_domain=*/r_domain,
            /*r_domain=*/l_domain,
            /*l_dim_ordering=*/r_dim_ordering,
            /*r_dim_ordering=*/l_dim_ordering));
  }

  ASSERT(differing_l_dim_size > differing_r_dim_size);

  ManyToOne<nonnegative_int, nonnegative_int> differing_l_to_r = [&]() {
    ASSERT(differing_l_dim_size % differing_r_dim_size == 0);

    positive_int scale_factor = positive_int{
        differing_l_dim_size / differing_r_dim_size,
    };

    return many_to_one_from_unstructured_relation(
        transform(set_of(nonnegative_range(differing_l_dim_size)),
                  [&](nonnegative_int l_entry)
                      -> std::pair<nonnegative_int, nonnegative_int> {
                    return std::pair{
                        l_entry,
                        l_entry / scale_factor,
                    };
                  }));
  }();

  std::unordered_set<L> nondiffering_l_dims =
      set_minus(input_dims_of_eq_projection(eq_proj),
                std::unordered_set{differing_l_dim});

  std::unordered_set<R> nondiffering_r_dims =
      set_minus(output_dims_of_eq_projection(eq_proj),
                std::unordered_set{differing_r_dim});

  DimDomainBiuniqueMapping<L, R> mapping_without_differing =
      dim_domain_biunique_mapping_from_projection(
          DimProjection<L, R>{
              EqProjection<L, R>{
                  with_matching_dim_sizes,
              },
          },
          restrict_domain_to_dims(l_domain, nondiffering_l_dims),
          restrict_domain_to_dims(r_domain, nondiffering_r_dims),
          l_dim_ordering,
          r_dim_ordering);

  auto l_coord_to_r_coord = [&](DimCoord<L> const &l_coord) -> DimCoord<R> {
    DimCoord<L> nondiffer_l_coord =
        restrict_coord_to_dims(l_coord, nondiffering_l_dims);
    nonnegative_int differ_l_coord = l_coord.raw.at(differing_l_dim);

    DimCoord<R> nondiffer_r_coord =
        mapping_without_differing.at_l(nondiffer_l_coord);
    nonnegative_int differ_r_coord = differing_l_to_r.at_l(differ_l_coord);

    std::unordered_map<R, nonnegative_int> raw_result =
        binary_merge_disjoint_maps(nondiffer_r_coord.raw,
                                   std::unordered_map<R, nonnegative_int>{
                                       {differing_r_dim, differ_r_coord},
                                   });

    return DimCoord<R>{
        raw_result,
    };
  };

  DimDomainHemiuniqueMapping<L, R> result = DimDomainHemiuniqueMapping<L, R>{
      /*coord_mapping=*/HemiuniqueBinaryRelation<DimCoord<L>, DimCoord<R>>{
          many_to_one_from_unstructured_relation(
              transform(get_coords_in_dim_domain(l_domain),
                        [&](DimCoord<L> const &l_coord)
                            -> std::pair<DimCoord<L>, DimCoord<R>> {
                          return std::pair{
                              l_coord,
                              l_coord_to_r_coord(l_coord),
                          };
                        })),

      },
      /*l_domain=*/l_domain,
      /*r_domain=*/r_domain,
  };

  return result;
}

/**
 * \brief Lower a \ref DimProjection to a \ref DimDomainHemiuniqueMapping.
 *
 * \relates DimDomainHemiuniqueMapping
 * \relates DimProjection
 */
template <typename L, typename R>
DimDomainHemiuniqueMapping<L, R> dim_domain_hemiunique_mapping_from_projection(
    DimProjection<L, R> const &projection,
    DimDomain<L> const &l_domain,
    DimDomain<R> const &r_domain,
    DimOrdering<L> const &l_dim_ordering,
    DimOrdering<R> const &r_dim_ordering) {
  if (dim_domain_volume(l_domain) == dim_domain_volume(r_domain)) {
    DimDomainBiuniqueMapping<L, R> biunique =
        dim_domain_biunique_mapping_from_projection(
            projection, l_domain, r_domain, l_dim_ordering, r_dim_ordering);

    return hemiunique_from_biunique_dim_domain_mapping(biunique);
  } else if (projection.is_eq_proj()) {
    // TODO(@lockshaw)(#pr):
    NOT_IMPLEMENTED();
  } else {
    PANIC("Not sure how to handle this case, but I don't think it should occur "
          "in practice. "
          "If you encounter this error, contact @lockshaw.");
  }
}

} // namespace FlexFlow

namespace nlohmann {

template <typename L, typename R>
struct adl_serializer<::FlexFlow::DimDomainHemiuniqueMapping<L, R>> {
  static void to_json(json &,
                      ::FlexFlow::DimDomainHemiuniqueMapping<L, R> const &) {
    // TODO(@lockshaw)(#pr):
    NOT_IMPLEMENTED();
  }

  static ::FlexFlow::DimDomainHemiuniqueMapping<L, R> from_json(json const &) {
    // TODO(@lockshaw)(#pr):
    NOT_IMPLEMENTED();
  }
};

} // namespace nlohmann

namespace std {

template <typename L, typename R>
struct hash<::FlexFlow::DimDomainHemiuniqueMapping<L, R>> {
  size_t operator()(::FlexFlow::DimDomainHemiuniqueMapping<L, R> const
                        &dim_domain_mapping) const {
    return get_std_hash(dim_domain_mapping.tie());
  }
};

} // namespace std
#endif
