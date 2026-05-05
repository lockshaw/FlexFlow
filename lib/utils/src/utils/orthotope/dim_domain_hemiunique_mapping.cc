#include "utils/orthotope/dim_domain_hemiunique_mapping.h"
#include "utils/archetypes/ordered_value_type.h"
#include "utils/archetypes/value_type.h"

using ::FlexFlow::ordered_value_type;
using L = ordered_value_type<0>;
using R = ordered_value_type<1>;

namespace FlexFlow {

template struct DimDomainHemiuniqueMapping<L, R>;

template std::string format_as(DimDomainHemiuniqueMapping<L, R> const &);

template std::ostream &operator<<(std::ostream &,
                                  DimDomainHemiuniqueMapping<L, R> const &);

template
  DimDomainHemiuniqueMapping<L, R> hemiunique_from_biunique_dim_domain_mapping(
      DimDomainBiuniqueMapping<L, R> const &);

template DimDomainHemiuniqueMapping<L, R>
    dim_domain_mapping_identity_map(DimDomain<L> const &,
                                    DimDomain<R> const &,
                                    DimOrdering<L> const &,
                                    DimOrdering<R> const &);

template DimDomainHemiuniqueMapping<R, L>
    invert_dim_domain_mapping(DimDomainHemiuniqueMapping<L, R> const &);

template
  DimDomainHemiuniqueMapping<L, R>
    dim_domain_hemiunique_mapping_by_scaling_projection(DimProjection<L, R> const &,
                                                        DimDomain<L> const &,
                                                        DimDomain<R> const &,
                                                        DimOrdering<L> const &,
                                                        DimOrdering<R> const &);

template DimDomainHemiuniqueMapping<L, R>
    dim_domain_hemiunique_mapping_from_projection(DimProjection<L, R> const &,
                                                  DimDomain<L> const &,
                                                  DimDomain<R> const &,
                                                  DimOrdering<L> const &,
                                                  DimOrdering<R> const &);

using T1 = value_type<2>;
using T2 = value_type<3>;
using T3 = value_type<4>;

template DimDomainHemiuniqueMapping<T1, T3>
    compose_dim_domain_mappings(DimDomainHemiuniqueMapping<T1, T2> const &,
                                DimDomainHemiuniqueMapping<T2, T3> const &);

} // namespace FlexFlow

namespace std {

template struct hash<::FlexFlow::DimDomainHemiuniqueMapping<L, R>>;

}
