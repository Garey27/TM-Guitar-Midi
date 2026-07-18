cmake_minimum_required(VERSION 3.24)

project(TM_Guitar_Midi VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(CTest)
include(FetchContent)

option(TMGM_BUILD_PLUGIN "Build the audio plug-in" ON)
option(TMGM_BUILD_TESTS "Build native and plug-in tests" ${BUILD_TESTING})
option(TMGM_ENABLE_CUDA "Build the optional native CUDA trainer" OFF)

if(WIN32 OR APPLE)
    set(_tmgm_default_vst3 ON)
else()
    set(_tmgm_default_vst3 OFF)
endif()
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(_tmgm_default_lv2 ON)
else()
    set(_tmgm_default_lv2 OFF)
endif()
option(TMGM_BUILD_VST3 "Build the VST3 format" ${_tmgm_default_vst3})
option(TMGM_BUILD_LV2 "Build the LV2 format" ${_tmgm_default_lv2})
unset(_tmgm_default_vst3)
unset(_tmgm_default_lv2)

set(TMGM_MODEL_PACKAGE
    "${CMAKE_CURRENT_SOURCE_DIR}/models/strict-cap16-v3"
    CACHE PATH "Path to the authenticated strict-cap16-v3 model package")
set(TMGM_JUCE_SOURCE_DIR "" CACHE PATH
    "Optional local JUCE 8.0.13 source tree (avoids a network fetch)")
get_filename_component(TMGM_MODEL_PACKAGE
    "${TMGM_MODEL_PACKAGE}" ABSOLUTE BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")

include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/VerifyModelPackage.cmake")
tmgm_verify_model_package("${TMGM_MODEL_PACKAGE}")

# Plug-in inference is CPU-only, but a root build may still opt into the
# independent CUDA trainer with -DTMGM_ENABLE_CUDA=ON.
set(BUILD_TESTING ${TMGM_BUILD_TESTS} CACHE BOOL
    "Build tests" FORCE)
add_subdirectory(native)

if(TMGM_BUILD_PLUGIN)
    if(NOT TMGM_BUILD_VST3 AND NOT TMGM_BUILD_LV2)
        message(FATAL_ERROR
            "TMGM_BUILD_PLUGIN is ON but no plug-in format is enabled")
    endif()
    if(TMGM_BUILD_LV2 AND NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
        message(FATAL_ERROR
            "The supported release matrix builds LV2 on Linux only")
    endif()

    # JUCE 8.0.13. CI fetches the pinned release tag; offline/local builds can
    # pass -DTMGM_JUCE_SOURCE_DIR=/path/to/JUCE.
    if(TMGM_JUCE_SOURCE_DIR)
        get_filename_component(TMGM_JUCE_SOURCE_DIR
            "${TMGM_JUCE_SOURCE_DIR}" ABSOLUTE
            BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
        if(NOT EXISTS "${TMGM_JUCE_SOURCE_DIR}/CMakeLists.txt")
            message(FATAL_ERROR
                "TMGM_JUCE_SOURCE_DIR is not a JUCE source tree: ${TMGM_JUCE_SOURCE_DIR}")
        endif()
        add_subdirectory("${TMGM_JUCE_SOURCE_DIR}"
            "${CMAKE_CURRENT_BINARY_DIR}/_deps/juce-build"
            EXCLUDE_FROM_ALL)
    else()
        FetchContent_Declare(JUCE
            GIT_REPOSITORY https://github.com/juce-framework/JUCE.git
            GIT_TAG 8.0.13
            GIT_SHALLOW TRUE
            GIT_PROGRESS TRUE
        )
        FetchContent_MakeAvailable(JUCE)
    endif()
    file(STRINGS "${JUCE_SOURCE_DIR}/CMakeLists.txt" _tmgm_juce_project_line
        REGEX "^project\\(JUCE VERSION 8\\.0\\.13 LANGUAGES C CXX\\)$")
    if(NOT _tmgm_juce_project_line)
        message(FATAL_ERROR
            "The pinned JUCE source does not identify itself as 8.0.13")
    endif()
    unset(_tmgm_juce_project_line)

    add_subdirectory(plugin)
endif()
