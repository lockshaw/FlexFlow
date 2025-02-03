#include "utils/containers/foldl.h"
#include <catch2/catch_test_macros.hpp>
#include <string>
#include <vector>

using namespace FlexFlow;


  TEST_CASE("foldl") {
    SECTION("product") {
      std::vector<int> container = {1, 2, 3, 4, 5};
      int result =
          foldl(container, 1, [](int acc, int elem) { return acc * elem; });
      int correct = 120;
      CHECK(result == correct);
    }

    SECTION("string concat") {
      std::vector<int> container = {1, 2, 3, 4, 5};
      std::string result =
          foldl(container, std::string(""), [](std::string acc, int elem) {
            return acc + std::to_string(elem);
          });
      std::string correct = "12345";
      CHECK(result == correct);
    }
  }

  TEST_CASE("foldl1") {
    SECTION("product") {
      std::vector<int> container = {1, 2, 3, 4, 5};
      int result =
          foldl1(container, [](int acc, int elem) { return acc * elem; });
      int correct = 120;
      CHECK(result == correct);
    }

    SECTION("string concat") {
      std::vector<std::string> container = {"1", "2", "3", "4", "5"};
      std::string result =
          foldl1(container,
                 [](std::string acc, std::string elem) { return acc + elem; });
      std::string correct = "12345";
      CHECK(result == correct);
    }
  }
