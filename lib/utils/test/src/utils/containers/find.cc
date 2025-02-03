#include "utils/containers/find.h"
#include "test/utils/doctest/check_without_stringify.h"
#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <set>
#include <unordered_set>
#include <vector>

using namespace FlexFlow;


  TEST_CASE("find") {

    SECTION("vector") {
      std::vector<int> v = {1, 2, 3, 3, 4, 5, 3};

      SECTION("element found") {
        CHECK_WITHOUT_STRINGIFY(find(v, 3) == std::find(v.begin(), v.end(), 3));
      }

      SECTION("element not found") {
        CHECK_WITHOUT_STRINGIFY(find(v, 6) == std::find(v.begin(), v.end(), 6));
      }

      SECTION("multiple occurrences of element") {
        CHECK_WITHOUT_STRINGIFY(find(v, 3) == std::find(v.begin(), v.end(), 3));
      }
    }

    SECTION("unordered_set") {
      std::unordered_set<int> s = {1, 2, 3, 4, 5};

      SECTION("element in container") {
        CHECK_WITHOUT_STRINGIFY(find(s, 3) == std::find(s.begin(), s.end(), 3));
      }

      SECTION("element not in container") {
        CHECK_WITHOUT_STRINGIFY(find(s, 6) == std::find(s.begin(), s.end(), 6));
      }
    }

    SECTION("set") {
      std::set<int> s = {1, 2, 3, 4, 5};

      SECTION("element in container") {
        CHECK_WITHOUT_STRINGIFY(find(s, 3) == std::find(s.begin(), s.end(), 3));
      }

      SECTION("element not in container") {
        CHECK_WITHOUT_STRINGIFY(find(s, 6) == std::find(s.begin(), s.end(), 6));
      }
    }
  }
