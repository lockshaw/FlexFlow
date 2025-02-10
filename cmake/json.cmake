include(aliasing)

if (FF_USE_EXTERNAL_JSON)
  find_package(nlohmann_json REQUIRED)

  alias_library(json nlohmann_json)
else()
  include(FetchContent)
  set(JSON_BuildTests OFF CACHE INTERNAL "")

  FetchContent_Declare(
    jsondl 
    URL https://github.com/nlohmann/json/releases/download/v3.11.2/json.tar.xz
    DOWNLOAD_EXTRACT_TIMESTAMP ON
  )
  FetchContent_MakeAvailable(jsondl)

  alias_library(json nlohmann_json::nlohmann_json)
endif()
