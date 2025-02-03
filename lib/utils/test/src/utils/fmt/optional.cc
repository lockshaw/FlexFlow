#include "utils/fmt/optional.h"
#include <catch2/catch_test_macros.hpp>

using namespace ::FlexFlow;


  TEST_CASE("fmt::to_string(std::optional<int>)") {
    SECTION("has value") {
      std::optional<int> input = 4;
      std::string result = fmt::to_string(input);
      std::string correct = "4";
      CHECK(result == correct);
    }

    SECTION("does not have value") {
      std::optional<int> input = std::nullopt;
      std::string result = fmt::to_string(input);
      std::string correct = "nullopt";
      CHECK(result == correct);
    }
  }
