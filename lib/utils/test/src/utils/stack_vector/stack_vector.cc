#include "utils/stack_vector/stack_vector.h"
#include "test/utils/doctest/fmt/vector.h"
#include "test/utils/rapidcheck.h"
#include <catch2/catch_test_macros.hpp>
#include <iterator>
#include <catch2/catch_template_test_macros.hpp>
#include <rapidcheck.h>
#include <rapidcheck/catch.h>

using namespace FlexFlow;


  TEMPLATE_TEST_CASE(
      "stack_vector<TestType, MAXSIZE>::push_back", "", int, double, char) {
    constexpr std::size_t MAXSIZE = 5;
    using StackVector = stack_vector<TestType, MAXSIZE>;
    StackVector vector;

    vector.push_back(10);
    std::vector<TestType> result = vector;
    std::vector<TestType> correct = {10};
    CHECK(result == correct);

    vector.push_back(20);
    correct = {10, 20};
    result = vector;
    CHECK(result == correct);
  }

  TEMPLATE_TEST_CASE(
      "stack_vector<TestType, MAXSIZE>::operator[]", "", int, double, char) {
    constexpr std::size_t MAXSIZE = 5;
    using StackVector = stack_vector<TestType, MAXSIZE>;
    StackVector vector;

    vector.push_back(10);
    vector.push_back(20);
    vector.push_back(30);

    CHECK(vector[0] == 10);
    CHECK(vector[1] == 20);
    CHECK(vector[2] == 30);
  }

  TEMPLATE_TEST_CASE("stack_vector<TestType, MAXSIZE>::size", "", int, double, char) {
    constexpr std::size_t MAXSIZE = 5;
    using StackVector = stack_vector<TestType, MAXSIZE>;
    StackVector vector;

    CHECK(vector.size() == 0);

    vector.push_back(10);
    CHECK(vector.size() == 1);

    vector.push_back(20);
    CHECK(vector.size() == 2);
  }

  TEMPLATE_TEST_CASE(
      "stack_vector<TestType, MAXSIZE>::operator==", "", int, double, char) {
    constexpr std::size_t MAXSIZE = 5;
    using StackVector = stack_vector<TestType, MAXSIZE>;
    StackVector vector1, vector2;

    vector1.push_back(10);
    vector1.push_back(15);
    vector1.push_back(20);

    vector2.push_back(10);
    vector2.push_back(15);
    vector2.push_back(20);

    CHECK(vector1 == vector2);
  }

  TEMPLATE_TEST_CASE("stack_vector<TestType, MAXSIZE>::back", "", int, double, char) {
    constexpr std::size_t MAXSIZE = 5;
    using StackVector = stack_vector<TestType, MAXSIZE>;
    StackVector vector;

    vector.push_back(10);
    CHECK(vector.back() == 10);

    vector.push_back(20);
    CHECK(vector.back() == 20);
  }

  TEMPLATE_TEST_CASE(
      "stack_vector<TestType, MAXSIZE> - check for size bound", "", int, double, char) {
    constexpr std::size_t MAXSIZE = 10;
    rc::prop("within bound", [&](stack_vector<TestType, MAXSIZE> v) {
      RC_ASSERT(v.size() <= MAXSIZE);
    });
  }
