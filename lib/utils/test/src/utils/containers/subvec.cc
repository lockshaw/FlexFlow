#include "utils/containers/subvec.h"
#include "test/utils/doctest/fmt/vector.h"
#include <catch2/catch_test_macros.hpp>
#include <vector>

using namespace FlexFlow;


  TEST_CASE("subvec") {
    std::vector<int> v = {1, 2, 3, 4, 5};

    SECTION("Basic subvector") {
      auto result = subvec(v, 1, 4);
      std::vector<int> correct = {2, 3, 4};
      CHECK(result == correct);
    }

    SECTION("From beginning to index") {
      auto result = subvec(v, std::nullopt, 3);
      std::vector<int> correct = {1, 2, 3};
      CHECK(result == correct);
    }

    SECTION("From index to end") {
      auto result = subvec(v, 2, std::nullopt);
      std::vector<int> correct = {3, 4, 5};
      CHECK(result == correct);
    }

    SECTION("All of the vector") {
      auto result = subvec(v, std::nullopt, std::nullopt);
      std::vector<int> correct = {1, 2, 3, 4, 5};
      CHECK(result == correct);
    }

    SECTION("Start greater than end") {
      auto result = subvec(v, 3, 1);
      std::vector<int> correct = {};
      CHECK(result == correct);
    }

    SECTION("Start equal to end") {
      auto result = subvec(v, 3, 3);
      std::vector<int> correct = {};
      CHECK(result == correct);
    }

    SECTION("Negative indices") {
      auto result = subvec(v, -3, -1);
      std::vector<int> correct = {3, 4};
      CHECK(result == correct);
    }

    SECTION("Upper index is out of bounds by 1") {
      CHECK_THROWS(subvec(v, 2, 6));
    }

    SECTION("Lower index is out of bounds by 1") {
      CHECK_THROWS(subvec(v, -6, 2));
    }
  }
