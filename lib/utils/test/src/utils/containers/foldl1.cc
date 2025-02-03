#include "utils/containers/foldl1.h"
#include <catch2/catch_test_macros.hpp>

using namespace ::FlexFlow;


  TEST_CASE("foldl1(std::vector<T>, F)") {
    auto concat = [](std::string const &accum, std::string const &s) {
      return accum + s;
    };

    SECTION("empty input") {
      std::vector<std::string> input = {};
      CHECK_THROWS(foldl1(input, concat));
    }

    SECTION("non-empty input") {
      std::vector<std::string> input = {"a s", "tr", "ing"};

      std::string result = foldl1(input, concat);

      std::string correct = "a string";

      CHECK(result == correct);
    }
  }
