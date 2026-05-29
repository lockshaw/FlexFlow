#include "utils/orthotope/dim_domain_biunique_mapping.h"
#include "utils/archetypes/value_type.h"

using ::FlexFlow::value_type;
using L = value_type<0>;
using R = value_type<1>;

namespace FlexFlow {

template struct DimDomainBiuniqueMapping<L, R>;

template std::string format_as(DimDomainBiuniqueMapping<L, R> const &);

template std::ostream &operator<<(std::ostream &,
                                  DimDomainBiuniqueMapping<L, R> const &);

template DimDomainBiuniqueMapping<L, R>
    dim_domain_biunique_mapping_identity_map(DimDomain<L> const &,
                                    DimDomain<R> const &,
                                    DimOrdering<L> const &,
                                    DimOrdering<R> const &);

template DimDomainBiuniqueMapping<L, R> empty_dim_domain_biunique_mapping();

template DimDomainBiuniqueMapping<R, L>
    invert_dim_domain_biunique_mapping(DimDomainBiuniqueMapping<L, R> const &);

template DimDomainBiuniqueMapping<L, R>
    dim_domain_biunique_mapping_from_projection(DimProjection<L, R> const &,
                                                DimDomain<L> const &,
                                                DimDomain<R> const &,
                                                DimOrdering<L> const &,
                                                DimOrdering<R> const &);

using T1 = value_type<2>;
using T2 = value_type<3>;
using T3 = value_type<4>;

template DimDomainBiuniqueMapping<T1, T3>
    compose_dim_domain_biunique_mappings(DimDomainBiuniqueMapping<T1, T2> const &,
                                DimDomainBiuniqueMapping<T2, T3> const &);

} // namespace FlexFlow

namespace std {

template struct hash<::FlexFlow::DimDomainBiuniqueMapping<L, R>>;

}
