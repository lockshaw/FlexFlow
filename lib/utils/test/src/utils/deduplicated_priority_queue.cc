#include "utils/deduplicated_priority_queue.h"
#include <catch2/catch_test_macros.hpp>

using namespace ::FlexFlow;


  TEST_CASE("DeduplicatedPriorityQueue push and pop") {
    DeduplicatedPriorityQueue<int> queue;

    SECTION("Push elements") {
      queue.push(5);
      queue.push(2);
      queue.push(7);
      queue.push(2);

      CHECK(queue.size() == 3);
      CHECK(queue.top() == 7);
      CHECK_FALSE(queue.empty());
    }

    SECTION("Pop elements") {
      queue.push(5);
      queue.push(2);
      queue.push(7);

      queue.pop();
      CHECK(queue.size() == 2);
      CHECK(queue.top() == 5);

      queue.pop();
      CHECK(queue.size() == 1);
      CHECK(queue.top() == 2);

      queue.pop();
      CHECK(queue.empty());
    }
  }
