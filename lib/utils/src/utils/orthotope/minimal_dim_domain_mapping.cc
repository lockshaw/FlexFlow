#include "utils/orthotope/minimal_dim_domain_mapping.h"
#include "utils/archetypes/jsonable_ordered_value_type.h"

using ::FlexFlow::jsonable_ordered_value_type;
using L = jsonable_ordered_value_type<0>;
using R = jsonable_ordered_value_type<1>;

namespace FlexFlow {

template struct MinimalDimDomainHemiuniqueMapping<L, R>;

template std::string format_as(MinimalDimDomainHemiuniqueMapping<L, R> const &);

template std::ostream &
    operator<<(std::ostream &, MinimalDimDomainHemiuniqueMapping<L, R> const &);

template MinimalDimDomainHemiuniqueMapping<L, R>
    minimal_hemiunique_mapping_from_dim_domain_mapping(
        DimDomainHemiuniqueMapping<L, R> const &);

template DimDomainHemiuniqueMapping<L, R>
    dim_domain_hemiunique_mapping_from_minimal_dim_domain(
        MinimalDimDomainHemiuniqueMapping<L, R> const &,
        std::set<L> const &,
        std::set<R> const &);

template MinimalDimDomainHemiuniqueMapping<L, R>
    minimal_dim_domain_hemiunique_mapping_identity_map(
        MinimalDimDomain<L> const &,
        MinimalDimDomain<R> const &,
        DimOrdering<L> const &,
        DimOrdering<R> const &);

template MinimalDimDomainHemiuniqueMapping<L, R>
    empty_minimal_dim_domain_mapping();

template MinimalDimDomainHemiuniqueMapping<R, L>
    invert_minimal_dim_domain_hemiunique_mapping(
        MinimalDimDomainHemiuniqueMapping<L, R> const &);

template MinimalDimDomainHemiuniqueMapping<L, R>
    minimal_dim_domain_hemiunique_mapping_from_projection(
        DimProjection<L, R> const &,
        MinimalDimDomain<L> const &,
        MinimalDimDomain<R> const &,
        DimOrdering<L> const &,
        DimOrdering<R> const &);

using T1 = jsonable_ordered_value_type<2>;
using T2 = jsonable_ordered_value_type<3>;
using T3 = jsonable_ordered_value_type<4>;

template MinimalDimDomainHemiuniqueMapping<T1, T3>
    compose_minimal_dim_domain_hemiunique_mappings(
        MinimalDimDomainHemiuniqueMapping<T1, T2> const &,
        MinimalDimDomainHemiuniqueMapping<T2, T3> const &);

template DimDomainHemiuniqueMapping<T1, T3>
    compose_dim_domain_hemiunique_mappings_through_minimal(
        DimDomainHemiuniqueMapping<T1, T2> const &,
        DimDomainHemiuniqueMapping<T2, T3> const &);

} // namespace FlexFlow

namespace std {

template struct hash<::FlexFlow::MinimalDimDomainHemiuniqueMapping<L, R>>;

}
