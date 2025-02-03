#include "utils/disjoint_set.h"
#include "test/utils/doctest/fmt/optional.h"
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>

using namespace FlexFlow;

template <typename T>
T generate_element(int seed);

template <>
int generate_element<int>(int seed) {
  return seed;
}

template <>
std::string generate_element<std::string>(int seed) {
  return "Element" + std::to_string(seed);
}


  TEMPLATE_TEST_CASE("DisjointSetUnionAndFind", "", int, std::string) {
    disjoint_set<std::optional<TestType>> ds;

    SECTION("SingleElementSets") {
      std::optional<TestType> element = generate_element<TestType>(1);
      CHECK(ds.find(element) == element);

      element = generate_element<TestType>(2);
      CHECK(ds.find(element) == element);
    }

    SECTION("UnionAndFind") {
      std::optional<TestType> element1 = generate_element<TestType>(1);
      std::optional<TestType> element2 = generate_element<TestType>(2);
      std::optional<TestType> element3 = generate_element<TestType>(3);
      std::optional<TestType> element4 = generate_element<TestType>(4);

      ds.m_union(element1, element2);
      CHECK(ds.find(element1) == ds.find(element2));

      ds.m_union(element3, element4);
      CHECK(ds.find(element3) == ds.find(element4));

      ds.m_union(element1, element3);
      CHECK(ds.find(element1) == ds.find(element3));
      CHECK(ds.find(element2) == ds.find(element4));
      CHECK(ds.find(element1) == ds.find(element2));
      CHECK(ds.find(element1) == ds.find(element4));
    }
  }

  TEST_CASE("DisjointSetMapping") {
    disjoint_set<int> ds;
    ds.m_union(1, 2);
    ds.m_union(3, 4);
    ds.m_union(1, 4);
    ds.m_union(5, 6);

    std::map<std::optional<int>, std::optional<int>, OptionalComparator<int>>
        expectedMapping = {{1, 4}, {2, 4}, {3, 4}, {4, 4}, {5, 6}, {6, 6}};

    std::map<std::optional<int>, std::optional<int>, OptionalComparator<int>>
        mapping = ds.get_mapping();

    for (auto const &kv : mapping) {
      CHECK(*kv.second == *expectedMapping[kv.first]); // Compare the values
                                                       // inside the optionals
    }
  }
