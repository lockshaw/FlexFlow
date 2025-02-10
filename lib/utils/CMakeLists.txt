ff_add_library(
  NAME
    utils
  SRC_PATTERNS
    src/*.cc
  PUBLIC_INCLUDE
    include/
  PRIVATE_INCLUDE
    src/
  DEPS
    expected
    fmt
    json
    cuda
)      

add_subdirectory(ffi)
add_subdirectory(test)
