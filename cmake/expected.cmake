include(aliasing)
if (FF_USE_EXTERNAL_EXPECTED)
  find_package(tl-expected REQUIRED)
  alias_library(expected tl::expected)
else()
  include(FetchContent)

  set(EXPECTED_BUILD_TESTS OFF)
  set(EXPECTED_BUILD_PACKAGE OFF)

  FetchContent_Declare(
    expecteddl 
    URL https://github.com/TartanLlama/expected/archive/refs/tags/v1.1.0.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP ON
  )
  FetchContent_MakeAvailable(expecteddl)
endif()
