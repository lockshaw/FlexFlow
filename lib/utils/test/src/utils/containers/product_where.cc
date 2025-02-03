#include "utils/containers/product_where.h"
#include <catch2/catch_test_macros.hpp>
#include <vector>

using namespace ::FlexFlow;



  TEST_CASE("product_where") {

    SECTION("empty starting container") {
      std::vector<int> input = {};
      auto condition = [](int x) { return x % 2 == 0; };
      int correct = 1;
      int result = product_where(input, condition);
      CHECK(correct == result);
    }

    SECTION("non-empty filtered container") {
      std::vector<int> input = {1, -2, 3, 4, 5};
      auto condition = [](int x) { return x % 2 == 0; };
      int correct = -8;
      int result = product_where(input, condition);
      CHECK(correct == result);
    }
    SECTION("empty filtered container") {
      std::vector<int> input = {1, 2, 3, 4, 5};
      auto condition = [](int x) { return x > 10; };
      int correct = 1;
      int result = product_where(input, condition);
      CHECK(correct == result);
    }
  }
