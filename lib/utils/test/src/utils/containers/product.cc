#include "utils/containers/product.h"
#include "utils/nonnegative_int/nonnegative_int.h"
#include <climits>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <set>
#include <unordered_set>
#include <vector>

using namespace ::FlexFlow;



  TEMPLATE_TEST_CASE("product",
                     "",
                     std::vector<int>,
                     std::vector<double>,
                     std::set<int>,
                     std::unordered_set<int>) {

    SECTION("non-empty container") {
      TestType input = {1, -2, 3, 5};
      auto correct = -30;
      auto result = product(input);
      CHECK(correct == result);
    }

    SECTION("empty container") {
      TestType input = {};
      auto correct = 1;
      auto result = product(input);
      CHECK(correct == result);
    }
  }

  TEST_CASE("product(std::vector<nonnegative_int>)") {
    SECTION("non-empty container") {
      std::vector<nonnegative_int> input = {1_n, 2_n, 3_n, 5_n};
      nonnegative_int correct = 30_n;
      auto result = product(input);
      CHECK(correct == result);
    }

    SECTION("empty container") {
      std::vector<nonnegative_int> input = {5_n};
      nonnegative_int correct = 5_n;
      // correct = nonnegative_int{x};
      // CHECK(x == 3);
      nonnegative_int result = product(input);
      CHECK(correct == correct);
    }
  }
