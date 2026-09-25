#include <doctest/doctest.h>
#include "utils/containers/take_until.h"
#include "utils/not_implemented.h"
#include "test/utils/doctest/fmt/vector.h"

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("take_until") {
    auto always_true = [](int) -> bool {
      return true; 
    };
    
    auto always_false = [](int) -> bool {
      return false; 
    };

    SUBCASE("input is empty") {
      std::vector<int> input = {};            

      std::vector<int> result = take_until(input, 
                                           [](int) -> bool {
                                             PANIC(); 
                                           });
      std::vector<int> correct = {};
    }

    SUBCASE("function never returns true") {
      std::vector<int> input = {1, 3, 2, 3};            

      std::vector<int> result = take_until(input, always_false);
      std::vector<int> correct = {1, 3, 2, 3};

      CHECK(result == correct);
    }

    SUBCASE("function immediately returns true") {
      std::vector<int> input = {1, 3, 2, 3};            

      std::vector<int> result = take_until(input, always_true);
      std::vector<int> correct = {};

      CHECK(result == correct);
    }

    SUBCASE("function eventually returns true") {
      std::vector<int> input = {1, 3, 2, 3};            

      std::vector<int> result = take_until(input, 
                                           [](int x) -> bool {
                                             return x % 2 == 0;
                                           });
      std::vector<int> correct = {1, 3};

      CHECK(result == correct);
    }
  }
}
