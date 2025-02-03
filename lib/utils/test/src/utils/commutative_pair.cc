#include "utils/commutative_pair.h"
#include "utils/containers/contains.h"
#include <catch2/catch_test_macros.hpp>
#include <rapidcheck.h>
#include <rapidcheck/catch.h>

using namespace ::FlexFlow;


  TEST_CASE("commutative_pair") {
    commutative_pair<int> x = {1, 2};
    commutative_pair<int> y = {2, 1};
    commutative_pair<int> z = {1, 1};

    SECTION("max and min") {
      SECTION("max") {
        CHECK(x.max() == 2);
      }

      SECTION("min") {
        CHECK(x.min() == 1);
      }

      rc::prop("max >= min", [](commutative_pair<int> const &p) {
        return p.max() >= p.min();
      });
    }

    SECTION("==") {
      CHECK(x == x);
      CHECK(x == y);
      CHECK_FALSE(x == z);

      rc::prop("== is reflexive",
                 [](commutative_pair<int> const &p) { return p == p; });
    }

    SECTION("!=") {
      CHECK_FALSE(x != x);
      CHECK_FALSE(x != y);
      CHECK(x != z);

      rc::prop("!= is anti-reflexive",
                 [](commutative_pair<int> const &p) { return !(p != p); });
    }

    SECTION("<") {
      CHECK_FALSE(x < x);
      CHECK_FALSE(x < y);
      CHECK(z < x);
      CHECK_FALSE(x < z);

      rc::prop("< uses left entry", [](int i1, int i2) {
        return commutative_pair<int>{i1, i2} <
               commutative_pair<int>{i1 + 1, i2};
      });

      rc::prop("< uses right entry", [](int i1, int i2) {
        return commutative_pair<int>{i1, i2} <
               commutative_pair<int>{i1, i2 + 1};
      });

      rc::prop(
          "< is antisymmetric",
          [](commutative_pair<int> const &p1, commutative_pair<int> const &p2) {
            RC_PRE(p1 < p2);
            return !(p2 < p1);
          });

      rc::prop("< is anti-reflexive",
                 [](commutative_pair<int> const &p) { return !(p < p); });

      rc::prop("< is transitive",
                 [](commutative_pair<int> const &p1,
                    commutative_pair<int> const &p2,
                    commutative_pair<int> const &p3) {
                   RC_PRE(p1 < p2 && p2 < p3);
                   return p1 < p3;
                 });
    }

    SECTION(">") {
      CHECK_FALSE(x > x);
      CHECK_FALSE(x > y);
      CHECK(x > z);

      rc::prop("> uses left entry", [](int i1, int i2) {
        return commutative_pair<int>{i1, i2} >
               commutative_pair<int>{i1 - 1, i2};
      });

      rc::prop("> uses right entry", [](int i1, int i2) {
        return commutative_pair<int>{i1, i2} >
               commutative_pair<int>{i1, i2 - 1};
      });

      rc::prop("> is antireflexive",
                 [](commutative_pair<int> const &p) { return !(p > p); });

      rc::prop(
          "> is antisymmetric",
          [](commutative_pair<int> const &p1, commutative_pair<int> const &p2) {
            RC_PRE(p1 > p2);
            return !(p2 > p1);
          });

      rc::prop("> is transitive",
                 [](commutative_pair<int> const &p1,
                    commutative_pair<int> const &p2,
                    commutative_pair<int> const &p3) {
                   RC_PRE(p1 < p2 && p2 < p3);
                   return p1 < p3;
                 });

      rc::prop(
          "< implies flipped >",
          [](commutative_pair<int> const &p1, commutative_pair<int> const &p2) {
            RC_PRE(p1 < p2);
            return p2 > p1;
          });
    }

    SECTION("<=") {
      rc::prop("<= is reflexive",
                 [](commutative_pair<int> const &p) { return p <= p; });

      rc::prop(
          "< implies <=",
          [](commutative_pair<int> const &p1, commutative_pair<int> const &p2) {
            RC_PRE(p1 < p2);
            return p1 <= p2;
          });
    }

    SECTION(">=") {
      rc::prop(">= is reflexive",
                 [](commutative_pair<int> const &p) { return p >= p; });

      rc::prop(
          "> implies >=",
          [](commutative_pair<int> const &p1, commutative_pair<int> const &p2) {
            RC_PRE(p1 > p2);
            return p1 >= p2;
          });
    }

    SECTION("std::hash") {
      CHECK(get_std_hash(x) == get_std_hash(x));
      CHECK(get_std_hash(x) != get_std_hash(z));
    }

    SECTION("fmt::to_string") {
      std::string result = fmt::to_string(x);
      std::unordered_set<std::string> correct_options = {"{2, 1}", "{1, 2}"};
      CHECK(contains(correct_options, result));
    }

    SECTION("operator<<") {
      std::ostringstream oss;
      oss << x;
      std::string result = oss.str();
      std::unordered_set<std::string> correct_options = {"{2, 1}", "{1, 2}"};
      CHECK(contains(correct_options, result));
    }
  }
