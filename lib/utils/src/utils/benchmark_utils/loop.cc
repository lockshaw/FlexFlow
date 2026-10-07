#include "utils/benchmark_utils/loop.h"

namespace {

void check_loop_builds(bool dry_run) {
  int x = 0;
  LOOP(10, dry_run) {
    x++;
    x++;
  }
}

} // namespace
