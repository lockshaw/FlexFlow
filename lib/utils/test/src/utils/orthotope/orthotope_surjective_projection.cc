#include "utils/orthotope/orthotope_surjective_projection.h"
#include "utils/containers/zip.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("deconflict_noninjective_dims") {
    SUBCASE("single input dim is unaffected") {
      std::vector<int> coords = {2};
      std::vector<int> dim_sizes = {5};

      int result = deconflict_noninjective_dims(zip(coords, dim_sizes));
      int correct = 2;

      CHECK(result == correct);
    }

    SUBCASE("basic example") {
      std::vector<int> coords = {4, 1};
      std::vector<int> dim_sizes = {5, 3};

      int result = deconflict_noninjective_dims(zip(coords, dim_sizes));
      int correct = 4 * 3 + 1;

      CHECK(result == correct);
    }

    SUBCASE("order matters") {
      std::vector<int> coords = {1, 4};
      std::vector<int> dim_sizes = {3, 5};

      int result = deconflict_noninjective_dims(zip(coords, dim_sizes));
      int correct = 1 * 5 + 4;

      CHECK(result == correct);
    }

    SUBCASE("throws if coord is outside of corresponding dim_size") {
      std::vector<int> coords = {2, 3, 1};
      std::vector<int> dim_sizes = {5, 3, 2};

      CHECK_THROWS(deconflict_noninjective_dims(zip(coords, dim_sizes)));
    }

    SUBCASE("throws if input is empty") {
      CHECK_THROWS(deconflict_noninjective_dims({}));
    }
  }
}
