include(aliasing)

if (FF_USE_EXTERNAL_FMT)
  find_package(fmt REQUIRED)
else()
  include(FetchContent)

  FetchContent_Declare(
    fmtdl 
    URL https://github.com/fmtlib/fmt/releases/download/10.1.1/fmt-10.1.1.zip
    DOWNLOAD_EXTRACT_TIMESTAMP ON
  )
  set(BUILD_SHARED_LIBS ON CACHE INTERNAL "Build SHARED libraries")
  set(CMAKE_POSITION_INDEPENDENT_CODE ON CACHE INTERNAL "")
  FetchContent_MakeAvailable(fmtdl)
endif()
alias_library(fmt fmt::fmt)
