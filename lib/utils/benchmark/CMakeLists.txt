ff_add_benchmarks(
  NAME
    utils
  SRC_PATTERNS
    src/*.cc
  PRIVATE_INCLUDE
    src/
  DEPS
    utils
)

add_subdirectory(common)
