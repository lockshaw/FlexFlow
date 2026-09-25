#include "utils/containers/binary_cartesian_product.h"
#include "test/utils/doctest/fmt/pair.h"
#include "test/utils/doctest/fmt/set.h"
#include "utils/hash/pair.h"
#include <doctest/doctest.h>
#include <string>
#include "utils/containers/set_of.h"

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("binary_cartesian_product") {
    auto mkp = [&](int l, std::string const &r) -> std::pair<int, std::string> {
      return std::pair<int, std::string>{l, r}; 
    };

    SUBCASE("both lhs and rhs are nonempty") {
      std::set<int> lhs = {1, 3};
      std::set<std::string> rhs = {"a", "b"};

      auto b = binary_cartesian_product(lhs, rhs);

      auto it = b.cbegin();
      ASSERT(*it == mkp(1, "a"));
      it++;
      ASSERT(*it == mkp(1, "b"));
      it++;
      ASSERT(*it == mkp(3, "a"));
      it++;
      ASSERT(*it == mkp(3, "b"));
      it++;
      ASSERT(it == b.cend());

      SUBCASE("full set conversion") {
        std::set<std::pair<int, std::string>> result =
            set_of(binary_cartesian_product(lhs, rhs));

        std::set<std::pair<int, std::string>> correct = {
            {1, "a"},
            {1, "b"},
            {3, "a"},
            {3, "b"},
        };

        CHECK(result == correct);
      }
    }

    SUBCASE("lhs is empty") {
      std::set<int> lhs = {};
      std::set<std::string> rhs = {"a", "b"};

      auto b = binary_cartesian_product(lhs, rhs);

      ASSERT(b.cbegin() == b.cend());

      SUBCASE("full set conversion") {
        std::set<std::pair<int, std::string>> result =
            set_of(binary_cartesian_product(lhs, rhs));

        std::set<std::pair<int, std::string>> correct = {};

        CHECK(result == correct);
      }
    }

    SUBCASE("rhs is empty") {
      std::set<int> lhs = {1, 2};
      std::set<std::string> rhs = {};

      auto b = binary_cartesian_product(lhs, rhs);

      ASSERT(b.cbegin() == b.cend());

      SUBCASE("full set conversion") {
        std::set<std::pair<int, std::string>> result =
            set_of(binary_cartesian_product(lhs, rhs));

        std::set<std::pair<int, std::string>> correct = {};

        CHECK(result == correct);
      }
    }

    SUBCASE("both lhs and rhs are empty") {
      std::set<int> lhs = {};
      std::set<std::string> rhs = {};

      auto b = binary_cartesian_product(lhs, rhs);

      ASSERT(b.cbegin() == b.cend());

      SUBCASE("full set conversion") {
        std::set<std::pair<int, std::string>> result =
            set_of(binary_cartesian_product(lhs, rhs));

        std::set<std::pair<int, std::string>> correct = {};

        CHECK(result == correct);
      }
    }
  }
}
