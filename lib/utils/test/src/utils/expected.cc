#include "utils/expected.h"
#include "test/utils/doctest/fmt/expected.h"
#include "test/utils/doctest/fmt/optional.h"
#include <catch2/catch_test_macros.hpp>

using namespace ::FlexFlow;


  TEST_CASE("optional_from_expected(tl::expected<T, E>)") {
    SECTION("has value") {
      tl::expected<int, std::string> input = 1;

      std::optional<int> result = optional_from_expected(input);
      std::optional<int> correct = 1;

      CHECK(result == correct);
    }

    SECTION("has unexpected") {
      tl::expected<int, std::string> input =
          tl::make_unexpected("error message");

      std::optional<int> result = optional_from_expected(input);
      std::optional<int> correct = std::nullopt;

      CHECK(result == correct);
    }
  }
