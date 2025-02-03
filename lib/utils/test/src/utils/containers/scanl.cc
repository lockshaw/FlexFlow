#include "utils/containers/scanl.h"
#include "test/utils/doctest/fmt/vector.h"
#include <catch2/catch_test_macros.hpp>
#include <string>
#include <vector>

using namespace FlexFlow;


  TEST_CASE("scanl") {
    SECTION("sum") {
      std::vector<int> input = {1, 2, 3, 4};
      std::vector<int> result =
          scanl(input, 0, [](int a, int b) { return a + b; });
      std::vector<int> correct = {0, 1, 3, 6, 10};
      CHECK(result == correct);
    }

    SECTION("custom function") {
      std::vector<int> input = {1, 3, 1, 2};
      auto op = [](int a, int b) { return (a + 1) * (b + 1); };
      std::vector<int> result = scanl(input, 1, op);
      std::vector<int> correct = {1, 4, 20, 42, 129};
      CHECK(result == correct);
    }

    SECTION("heterogeneous types") {
      std::vector<int> input = {1, 2, 3, 4};
      auto op = [](std::string const &a, int b) {
        return a + std::to_string(b);
      };
      std::vector<std::string> result = scanl(input, std::string(""), op);
      std::vector<std::string> correct = {"", "1", "12", "123", "1234"};
      CHECK(result == correct);
    }

    SECTION("empty input") {
      std::vector<int> input = {};
      std::vector<int> result =
          scanl(input, 0, [](int a, int b) { return a + b; });
      std::vector<int> correct = {0};
      CHECK(result == correct);
    }
  }

  TEST_CASE("scanl1") {
    SECTION("sum") {
      std::vector<int> input = {1, 2, 3, 4};
      std::vector<int> result =
          scanl1(input, [](int a, int b) { return a + b; });
      std::vector<int> correct = {1, 3, 6, 10};
      CHECK(result == correct);
    }

    SECTION("custom function") {
      std::vector<int> input = {1, 2, 5, 2};
      auto op = [](int a, int b) { return a * b + 1; };
      std::vector<int> result = scanl1(input, op);
      std::vector<int> correct = {1, 3, 16, 33};
      CHECK(result == correct);
    }

    SECTION("empty input") {
      std::vector<int> input = {};
      std::vector<int> result =
          scanl1(input, [](int a, int b) { return a + b; });
      std::vector<int> correct = {};
      CHECK(result == correct);
    }
  }
