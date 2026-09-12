#include "utils/orthotope/dim_coord.h"
#include "test/utils/doctest/fmt/set.h"
#include "test/utils/doctest/fmt/optional.h"
#include "utils/orthotope/dim_ordering.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("lift_dim_coord") {
    DimCoord<int> coord = DimCoord<int>{{
        {7, 0_n},
        {3, 4_n},
        {1, 1_n},
    }};

    SUBCASE("lifted dims are a superset of coord dims") {
      std::set<int> lifted_dims = {1, 3, 6, 7};

      DimCoord<int> result = lift_dim_coord(coord, lifted_dims);
      DimCoord<int> correct = DimCoord<int>{{
          {7, 0_n},
          {3, 4_n},
          {1, 1_n},
          {6, 0_n},
      }};

      CHECK(result == correct);
    }

    SUBCASE("lifted dims are the same as coord dims") {
      std::set<int> lifted_dims = {1, 3, 7};

      DimCoord<int> result = lift_dim_coord(coord, lifted_dims);
      DimCoord<int> correct = coord;

      CHECK(result == correct);
    }

    SUBCASE("lifted dims are a subset of coord dims") {
      std::set<int> lifted_dims = {1, 7};

      CHECK_THROWS(lift_dim_coord(coord, lifted_dims));
    }

    SUBCASE("lifted dims are overlapping with coord dims") {
      std::set<int> lifted_dims = {1, 2, 7};

      CHECK_THROWS(lift_dim_coord(coord, lifted_dims));
    }
  }

  TEST_CASE("flatten_dim_coord") {
    DimCoord<int> coord = DimCoord<int>{{
        {7, 0_n},
        {3, 4_n},
        {1, 1_n},
    }};

    DimDomain<int> domain = DimDomain<int>{{
        {7, 2_p},
        {3, 5_p},
        {1, 3_p},
    }};

    DimOrdering<int> dim_ordering = make_dim_ordering_from_vector<int>({
        3,
        7,
        1,
    });

    nonnegative_int result = flatten_dim_coord(coord, domain, dim_ordering);
    nonnegative_int correct = nonnegative_int{4 * 2 * 3 + 0 * 3 + 1};

    CHECK(result == correct);
  }

  TEST_CASE("unflatten_dim_coord") {
    DimDomain<int> domain = DimDomain<int>{{
        {7, 2_p},
        {3, 5_p},
        {1, 3_p},
    }};
    nonnegative_int flattened = nonnegative_int{4 * 2 * 3 + 0 * 3 + 1};

    DimOrdering<int> dim_ordering = make_dim_ordering_from_vector<int>({
        3,
        7,
        1,
    });

    DimCoord<int> result = unflatten_dim_coord(flattened, domain, dim_ordering);
    DimCoord<int> correct = DimCoord<int>{{
        {7, 0_n},
        {3, 4_n},
        {1, 1_n},
    }};

    CHECK(result == correct);
  }

  TEST_CASE("get_coords_in_dim_domain") {
    SUBCASE("one-dimensional dim domain") {
      DimDomain<int> dim_domain = DimDomain<int>{{
          {7, 2_p},
      }};

      std::set<DimCoord<int>> result = get_coords_in_dim_domain(dim_domain);

      std::set<DimCoord<int>> correct = {
          DimCoord<int>{{
              {7, 0_n},
          }},
          DimCoord<int>{{
              {7, 1_n},
          }},
      };

      CHECK(result == correct);
    }

    SUBCASE("multi-dimensional dim domain") {
      DimDomain<int> dim_domain = DimDomain<int>{{
          {7, 2_p},
          {2, 3_p},
      }};

      std::set<DimCoord<int>> result = get_coords_in_dim_domain(dim_domain);

      auto mk_dim_coord = [](nonnegative_int dim7, nonnegative_int dim2) {
        return DimCoord<int>{{
            {7, dim7},
            {2, dim2},
        }};
      };

      std::set<DimCoord<int>> correct = {
          mk_dim_coord(0_n, 0_n),
          mk_dim_coord(0_n, 1_n),
          mk_dim_coord(0_n, 2_n),
          mk_dim_coord(1_n, 0_n),
          mk_dim_coord(1_n, 1_n),
          mk_dim_coord(1_n, 2_n),
      };

      CHECK(result == correct);
    }

    SUBCASE("includes trivial dimension") {
      DimDomain<int> dim_domain = DimDomain<int>{{
          {7, 1_p},
          {2, 3_p},
      }};

      std::set<DimCoord<int>> result = get_coords_in_dim_domain(dim_domain);

      auto mk_dim_coord = [](nonnegative_int dim7, nonnegative_int dim2) {
        return DimCoord<int>{{
            {7, dim7},
            {2, dim2},
        }};
      };

      std::set<DimCoord<int>> correct = {
          mk_dim_coord(0_n, 0_n),
          mk_dim_coord(0_n, 1_n),
          mk_dim_coord(0_n, 2_n),
      };

      CHECK(result == correct);
    }

    SUBCASE("zero-dimensional dim domain") {
      DimDomain<int> dim_domain = DimDomain<int>{{}};

      std::set<DimCoord<int>> result = get_coords_in_dim_domain(dim_domain);

      std::set<DimCoord<int>> correct = {
          DimCoord<int>{{}},
      };

      CHECK(result == correct);
    }
  }

  TEST_CASE("smallest_dim_domain_for_coord_set") {
    SUBCASE("coord set is empty") {
      std::set<DimCoord<int>> coord_set = {};

      DimDomain<int> result = smallest_dim_domain_for_coord_set(coord_set);
      DimDomain<int> correct = empty_dim_domain<int>();

      CHECK(result == correct);
    }

    SUBCASE("coord set contains coords of different dimensionalities") {
      std::set<DimCoord<int>> coord_set = {
        DimCoord<int>{{
          {2, 1_n},
        }},
        DimCoord<int>{{
          {1, 2_n},
          {2, 3_n},
        }},
      };

      CHECK_THROWS(smallest_dim_domain_for_coord_set(coord_set));
    }

    SUBCASE("coord set contains only 0-d coords") {
      std::set<DimCoord<int>> coord_set = {
        DimCoord<int>{{}},
      };

      DimDomain<int> result = smallest_dim_domain_for_coord_set(coord_set);
      DimDomain<int> correct = empty_dim_domain<int>();

      CHECK(result == correct);
    }

    SUBCASE("correct usage") {
      auto coord = [](nonnegative_int x,
                      nonnegative_int y)
        -> DimCoord<int>
      {
        return DimCoord<int>{{
          {2, x},
          {-6, y},
        }};
      };

      std::set<DimCoord<int>> coord_set = {
        coord(0_n, 1_n),
        coord(2_n, 4_n),
        coord(3_n, 0_n),
      };

      DimDomain<int> result = smallest_dim_domain_for_coord_set(coord_set);
      DimDomain<int> correct = DimDomain<int>{{
        {2, 4_p},
        {-6, 5_p},
      }};

      CHECK(result == correct);
    }
  }

  TEST_CASE("strict_dim_domain_for_coord_set") {
    SUBCASE("coord set is empty") {
      std::set<DimCoord<int>> coord_set = {};

      std::optional<DimDomain<int>> result = strict_dim_domain_for_coord_set(coord_set);
      std::optional<DimDomain<int>> correct = std::nullopt;

      CHECK(result == correct);
    }

    SUBCASE("coord set contains coords of different dimensionalities") {
      std::set<DimCoord<int>> coord_set = {
        DimCoord<int>{{
          {2, 1_n},
        }},
        DimCoord<int>{{
          {1, 2_n},
          {2, 3_n},
        }},
      };

      CHECK_THROWS(strict_dim_domain_for_coord_set(coord_set));
    }

    SUBCASE("coord set contains only 0-d coords") {
      std::set<DimCoord<int>> coord_set = {
        DimCoord<int>{{}},
      };

      std::optional<DimDomain<int>> result = strict_dim_domain_for_coord_set(coord_set);
      std::optional<DimDomain<int>> correct = empty_dim_domain<int>();

      CHECK(result == correct);
    }

    auto coord = [](nonnegative_int x,
                    nonnegative_int y)
      -> DimCoord<int>
    {
      return DimCoord<int>{{
        {2, x},
        {-6, y},
      }};
    };

    SUBCASE("coord set is not orthotopic") {
      std::set<DimCoord<int>> coord_set = {
        coord(0_n, 1_n),
        coord(2_n, 4_n),
        coord(3_n, 0_n),
      };

      std::optional<DimDomain<int>> result = strict_dim_domain_for_coord_set(coord_set);
      std::optional<DimDomain<int>> correct = std::nullopt;

      CHECK(result == correct);
    }

    SUBCASE("coord set is orthotopic") {
      std::set<DimCoord<int>> coord_set = {
        coord(0_n, 0_n),
        coord(1_n, 0_n),
        coord(2_n, 0_n),
        coord(0_n, 1_n),
        coord(1_n, 1_n),
        coord(2_n, 1_n),
      };

      std::optional<DimDomain<int>> result = strict_dim_domain_for_coord_set(coord_set);
      std::optional<DimDomain<int>> correct = DimDomain<int>{{
        {2, 3_p},
        {-6, 2_p},
      }};

      CHECK(result == correct);
    }
  }
}
