#include <doctest/doctest.h>
#include "utils/containers/flatmap.h"

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("flatmap(std::string, F)") {
    std::string input = "aBabcBc";

    SUBCASE("replacement length > 1") {
      std::string result = flatmap(input, [](char c) -> std::string {
        if (c == 'B') {
          return "..";
        } else {
          return std::string{c};
        }
      });

      std::string correct = "a..abc..c";

      CHECK(result == correct);
    }

    SUBCASE("replacement length == 0") {
      std::string result = flatmap(input, [](char c) -> std::string {
        if (c == 'B') {
          return "";
        } else {
          return std::string{c};
        }
      });

      std::string correct = "aabcc";

      CHECK(result == correct);
    }
  }
}
