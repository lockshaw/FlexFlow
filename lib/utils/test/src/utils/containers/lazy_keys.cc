#include "utils/containers/lazy_keys.h"
#include "test/utils/doctest/check_without_stringify.h"
#include <doctest/doctest.h>
#include <string>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("lazy_keys") {
    SUBCASE("map is empty") {
      std::map<int, std::string> m = {};

      auto ks = lazy_keys(m);

      auto it = ks.begin();
      CHECK_WITHOUT_STRINGIFY(it == ks.end());
    }

    SUBCASE("map is nonempty") {
      std::map<int, std::string> m = {{1, "one"}, {2, "two"}, {3, "three"}};

      auto ks = lazy_keys(m);

      auto it = ks.begin();
      CHECK(*it == 1);
      it++;
      CHECK(*it == 2);
      it++;
      CHECK(*it == 3);
      it++;
      CHECK_WITHOUT_STRINGIFY(it == ks.end());
    }
  }
}
