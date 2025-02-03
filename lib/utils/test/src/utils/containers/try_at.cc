#include "utils/containers/try_at.h"
#include "test/utils/doctest/fmt/optional.h"
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <string>

using namespace ::FlexFlow;


  TEMPLATE_TEST_CASE("try_at(TestType, K)",
                     "",
                     (std::unordered_map<int, std::string>),
                     (std::map<int, std::string>)) {
    TestType m = {{1, "one"}, {2, "two"}};

    SECTION("map contains key") {
      std::optional<std::string> result = try_at(m, 1);
      std::optional<std::string> correct = "one";

      CHECK(result == correct);
    }

    SECTION("map does not contain key") {
      std::optional<std::string> result = try_at(m, 3);
      std::optional<std::string> correct = std::nullopt;

      CHECK(result == correct);
    }
  }
