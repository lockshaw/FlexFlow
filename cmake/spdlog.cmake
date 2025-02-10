include(aliasing)

if (FF_USE_EXTERNAL_SPDLOG)
  find_package(spdlog REQUIRED)
else()
  include(FetchContent)

  FetchContent_Declare(
    spdlogdl 
    URL https://github.com/gabime/spdlog/archive/refs/tags/v1.12.0.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP ON
  )
  FetchContent_MakeAvailable(spdlogdl)
endif()

add_library(ff_spdlog INTERFACE)
target_link_libraries(ff_spdlog INTERFACE spdlog::spdlog)
target_compile_definitions(ff_spdlog INTERFACE SPDLOG_FMT_EXTERNAL)
alias_library(spdlog ff_spdlog)
