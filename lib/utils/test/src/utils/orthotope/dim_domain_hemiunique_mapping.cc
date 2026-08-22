#include "utils/orthotope/dim_domain_hemiunique_mapping.h"
#include "utils/orthotope/dim_ordering.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("dim_domain_hemiunique_mapping_by_scaling_projection") {
    SUBCASE("scaling up an already parallel dimension") {
      DimDomain<int> input_domain = DimDomain<int>{{
          {1, 2_p},
          {3, 4_p},
          {7, 3_p},
          {8, 1_p},
      }};

      std::string dim_a = "a";
      std::string dim_b = "b";
      std::string dim_c = "c";
      std::string dim_d = "d";

      DimDomain<std::string> output_domain = DimDomain<std::string>{{
          {dim_a, 2_p},
          {dim_b, 2_p},
          {dim_c, 3_p},
          {dim_d, 1_p},
      }};

      DimOrdering<int> input_dim_ordering = make_dim_ordering_from_vector<int>({
          1,
          3,
          7,
          8,
      });

      DimOrdering<std::string> output_dim_ordering =
          make_dim_ordering_from_vector<std::string>({
              dim_a,
              dim_b,
              dim_c,
              dim_d,
          });

      DimProjection<int, std::string> projection = DimProjection{
          EqProjection{
              bidict<int, std::string>{
                  {1, "a"},
                  {3, "b"},
                  {7, "c"},
                  {8, "d"},
              },
          },
      };

      DimDomainHemiuniqueMapping<int, std::string> result =
          dim_domain_hemiunique_mapping_by_scaling_projection(
              /*projection=*/projection,
              /*l_domain=*/input_domain,
              /*r_domain=*/output_domain,
              /*l_dim_ordering=*/input_dim_ordering,
              /*r_dim_ordering=*/output_dim_ordering);

      auto mk_input_coord = [&](int dim1, int dim3, int dim7, int dim8) {
        ASSERT(dim1 < input_domain.dims.at(1));
        ASSERT(dim3 < input_domain.dims.at(3));
        ASSERT(dim7 < input_domain.dims.at(7));
        ASSERT(dim8 < input_domain.dims.at(8));

        return DimCoord<int>{{
            {1, nonnegative_int{dim1}},
            {3, nonnegative_int{dim3}},
            {7, nonnegative_int{dim7}},
            {8, nonnegative_int{dim8}},
        }};
      };

      auto mk_output_coord = [&](int a, int b, int c, int d) {
        ASSERT(a < output_domain.dims.at("a"));
        ASSERT(b < output_domain.dims.at("b"));
        ASSERT(c < output_domain.dims.at("c"));
        ASSERT(d < output_domain.dims.at("d"));

        return DimCoord<std::string>{{
            {dim_a, nonnegative_int{a}},
            {dim_b, nonnegative_int{b}},
            {dim_c, nonnegative_int{c}},
            {dim_d, nonnegative_int{d}},
        }};
      };

      DimDomainHemiuniqueMapping<int, std::string> correct =
          DimDomainHemiuniqueMapping<int, std::string>{
              /*coord_mapping=*/HemiuniqueBinaryRelation<DimCoord<int>,
                                                         DimCoord<std::string>>{
                  many_to_one_from_unstructured_relation(std::set<
                                                         std::pair<
                                                             DimCoord<int>,
                                                             DimCoord<
                                                                 std::string>>>{
                      {mk_input_coord(0, 0, 0, 0), mk_output_coord(0, 0, 0, 0)},
                      {mk_input_coord(0, 1, 0, 0), mk_output_coord(0, 0, 0, 0)},
                      {mk_input_coord(0, 0, 1, 0), mk_output_coord(0, 0, 1, 0)},
                      {mk_input_coord(0, 1, 1, 0), mk_output_coord(0, 0, 1, 0)},
                      {mk_input_coord(0, 0, 2, 0), mk_output_coord(0, 0, 2, 0)},
                      {mk_input_coord(0, 1, 2, 0), mk_output_coord(0, 0, 2, 0)},
                      {mk_input_coord(0, 2, 0, 0), mk_output_coord(0, 1, 0, 0)},
                      {mk_input_coord(0, 3, 0, 0), mk_output_coord(0, 1, 0, 0)},
                      {mk_input_coord(0, 2, 1, 0), mk_output_coord(0, 1, 1, 0)},
                      {mk_input_coord(0, 3, 1, 0), mk_output_coord(0, 1, 1, 0)},
                      {mk_input_coord(0, 2, 2, 0), mk_output_coord(0, 1, 2, 0)},
                      {mk_input_coord(0, 3, 2, 0), mk_output_coord(0, 1, 2, 0)},
                      {mk_input_coord(1, 0, 0, 0), mk_output_coord(1, 0, 0, 0)},
                      {mk_input_coord(1, 1, 0, 0), mk_output_coord(1, 0, 0, 0)},
                      {mk_input_coord(1, 0, 1, 0), mk_output_coord(1, 0, 1, 0)},
                      {mk_input_coord(1, 1, 1, 0), mk_output_coord(1, 0, 1, 0)},
                      {mk_input_coord(1, 0, 2, 0), mk_output_coord(1, 0, 2, 0)},
                      {mk_input_coord(1, 1, 2, 0), mk_output_coord(1, 0, 2, 0)},
                      {mk_input_coord(1, 2, 0, 0), mk_output_coord(1, 1, 0, 0)},
                      {mk_input_coord(1, 3, 0, 0), mk_output_coord(1, 1, 0, 0)},
                      {mk_input_coord(1, 2, 1, 0), mk_output_coord(1, 1, 1, 0)},
                      {mk_input_coord(1, 3, 1, 0), mk_output_coord(1, 1, 1, 0)},
                      {mk_input_coord(1, 2, 2, 0), mk_output_coord(1, 1, 2, 0)},
                      {mk_input_coord(1, 3, 2, 0), mk_output_coord(1, 1, 2, 0)},
                  }),
              },
              /*l_domain=*/input_domain,
              /*r_domain=*/output_domain,
          };

      CHECK(result == correct);
    }

    // TODO(@lockshaw)(#pr):
    // SUBCASE("adding a new parallel dimension") {
    //   DimDomain<int> input_domain = DimDomain<int>{{
    //       {1, 2_p},
    //       {2, 1_p},
    //   }};

    //   std::string dim_a = "a";
    //   std::string dim_b = "b";

    //   DimDomain<std::string> output_domain = DimDomain<std::string>{{
    //       {dim_a, 2_p},
    //       {dim_b, 2_p},
    //   }};

    //   DimOrdering<int> input_dim_ordering = make_dim_ordering_from_vector<int>({
    //       1,
    //   });

    //   DimOrdering<std::string> output_dim_ordering =
    //       make_dim_ordering_from_vector<std::string>({
    //           dim_a,
    //           dim_b,
    //       });

    //   DimProjection<int, std::string> projection = DimProjection{
    //       UpProjection{
    //           OneToMany<int, std::string>{
    //               {1, {dim_a, dim_b}},
    //           },
    //       },
    //   };

    //   DimDomainHemiuniqueMapping<int, std::string> result =
    //       dim_domain_hemiunique_mapping_by_scaling_projection(
    //           /*projection=*/projection,
    //           /*l_domain=*/input_domain,
    //           /*r_domain=*/output_domain,
    //           /*l_dim_ordering=*/input_dim_ordering,
    //           /*r_dim_ordering=*/output_dim_ordering);

    //   auto mk_input_coord = [&](int dim1) {
    //     ASSERT(dim1 < input_domain.dims.at(1));

    //     return DimCoord<int>{{
    //         {1, nonnegative_int{dim1}},
    //     }};
    //   };

    //   auto mk_output_coord = [&](int a, int b) {
    //     ASSERT(a < output_domain.dims.at("a"));
    //     ASSERT(b < output_domain.dims.at("b"));

    //     return DimCoord<std::string>{{
    //         {dim_a, nonnegative_int{a}},
    //         {dim_b, nonnegative_int{b}},
    //     }};
    //   };

    //   DimDomainHemiuniqueMapping<int, std::string> correct =
    //       DimDomainHemiuniqueMapping<int, std::string>{
    //           /*coord_mapping=*/HemiuniqueBinaryRelation<DimCoord<int>,
    //                                                      DimCoord<std::string>>{
    //               many_to_one_from_unstructured_relation(std::set<
    //                                                      std::pair<
    //                                                          DimCoord<int>,
    //                                                          DimCoord<
    //                                                              std::string>>>{
    //                   {mk_input_coord(0), mk_output_coord(0, 0)},
    //                   {mk_input_coord(0), mk_output_coord(0, 1)},
    //                   {mk_input_coord(1), mk_output_coord(1, 1)},
    //                   {mk_input_coord(1), mk_output_coord(1, 1)},
    //               }),
    //           },
    //           /*l_domain=*/input_domain,
    //           /*r_domain=*/output_domain,
    //       };

    //   CHECK(result == correct);
    // }
  }
}
