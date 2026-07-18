cmake_minimum_required(VERSION 3.24)

project(TMGMNative VERSION 0.1.0 LANGUAGES CXX)

option(TMGM_ENABLE_CUDA "Build the optional CUDA backend" ON)
set(TMGM_CUDA_ARCHITECTURES "native" CACHE STRING
    "CUDA architectures passed to CMAKE_CUDA_ARCHITECTURES")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(tmgm_dataset STATIC
    src/dataset.cpp
)
target_include_directories(tmgm_dataset PUBLIC include)
target_compile_features(tmgm_dataset PUBLIC cxx_std_17)

if(MSVC)
    target_compile_options(tmgm_dataset PRIVATE /W4 /permissive-)
else()
    target_compile_options(tmgm_dataset PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_library(tmgm_tm_reference STATIC
    src/tm_reference.cpp
)
target_include_directories(tmgm_tm_reference PUBLIC include)
target_compile_features(tmgm_tm_reference PUBLIC cxx_std_17)
if(MSVC)
    target_compile_options(tmgm_tm_reference PRIVATE /W4 /permissive-)
else()
    target_compile_options(tmgm_tm_reference PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_library(tmgm_model STATIC
    src/model.cpp
)
target_include_directories(tmgm_model PUBLIC include)
target_compile_features(tmgm_model PUBLIC cxx_std_17)
if(MSVC)
    target_compile_options(tmgm_model PRIVATE /W4 /permissive-)
else()
    target_compile_options(tmgm_model PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_library(tmgm_ensemble STATIC
    src/ensemble_bundle.cpp
    src/ensemble_inference.cpp
    src/strict_cap16_v3.cpp
)
target_include_directories(tmgm_ensemble PUBLIC include)
target_compile_features(tmgm_ensemble PUBLIC cxx_std_17)
target_link_libraries(tmgm_ensemble PUBLIC tmgm_dataset)
if(MSVC)
    target_compile_options(tmgm_ensemble PRIVATE /W4 /permissive- /fp:strict)
else()
    target_compile_options(tmgm_ensemble PRIVATE
        -Wall -Wextra -Wpedantic -ffp-contract=off)
endif()

add_library(tmgm_frontend STATIC
    src/strict_cap16_v3_frontend.cpp
)
target_include_directories(tmgm_frontend PUBLIC include)
target_include_directories(tmgm_frontend PRIVATE third_party/pocketfft)
target_compile_features(tmgm_frontend PUBLIC cxx_std_17)
target_link_libraries(tmgm_frontend PUBLIC tmgm_ensemble)
if(MSVC)
    target_compile_options(tmgm_frontend PRIVATE /W4 /permissive- /fp:strict)
else()
    target_compile_options(tmgm_frontend PRIVATE
        -Wall -Wextra -Wpedantic -ffp-contract=off)
endif()

add_library(tmgm_resampler STATIC
    src/causal_resampler.cpp
)
target_include_directories(tmgm_resampler PUBLIC include)
target_compile_features(tmgm_resampler PUBLIC cxx_std_17)
if(MSVC)
    target_compile_options(tmgm_resampler PRIVATE
        /W4 /permissive- /fp:strict)
else()
    target_compile_options(tmgm_resampler PRIVATE
        -Wall -Wextra -Wpedantic -ffp-contract=off)
endif()

add_library(tmgm_tm_calibration STATIC
    src/tm_calibration.cpp
)
target_include_directories(tmgm_tm_calibration PUBLIC include)
target_compile_features(tmgm_tm_calibration PUBLIC cxx_std_17)
target_link_libraries(tmgm_tm_calibration PUBLIC tmgm_dataset tmgm_model)
if(MSVC)
    target_compile_options(tmgm_tm_calibration PRIVATE /W4 /permissive-)
else()
    target_compile_options(tmgm_tm_calibration PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_library(tmgm_tm_validation STATIC
    src/tm_validation.cpp
)
target_include_directories(tmgm_tm_validation PUBLIC include)
target_compile_features(tmgm_tm_validation PUBLIC cxx_std_17)
target_link_libraries(tmgm_tm_validation PUBLIC tmgm_dataset)
if(MSVC)
    target_compile_options(tmgm_tm_validation PRIVATE /W4 /permissive-)
else()
    target_compile_options(tmgm_tm_validation PRIVATE -Wall -Wextra -Wpedantic)
endif()

set(TMGM_HAS_CUDA OFF)
if(TMGM_ENABLE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        set(TMGM_HAS_CUDA ON)

        add_library(tmgm_cuda STATIC
            src/cuda_smoke.cu
            src/tm_cuda.cu
        )
        target_include_directories(tmgm_cuda PUBLIC include)
        target_compile_features(tmgm_cuda PUBLIC cxx_std_17)
        target_link_libraries(tmgm_cuda PUBLIC
            tmgm_dataset tmgm_model tmgm_tm_validation)
        set_target_properties(tmgm_cuda PROPERTIES
            CUDA_STANDARD 17
            CUDA_STANDARD_REQUIRED ON
            CUDA_ARCHITECTURES "${TMGM_CUDA_ARCHITECTURES}"
            CUDA_RUNTIME_LIBRARY Shared
        )
    endif()
endif()

if(NOT TMGM_HAS_CUDA)
    add_library(tmgm_cuda STATIC src/cuda_smoke_stub.cpp)
    target_include_directories(tmgm_cuda PUBLIC include)
    target_compile_features(tmgm_cuda PUBLIC cxx_std_17)
endif()

add_executable(tmgm_dataset_info src/main.cpp)
target_link_libraries(tmgm_dataset_info PRIVATE tmgm_dataset tmgm_cuda)
if(TMGM_HAS_CUDA)
    set_property(TARGET tmgm_dataset_info PROPERTY CUDA_RUNTIME_LIBRARY Shared)
endif()

add_executable(tmgm_ensemble_predict src/ensemble_predict_main.cpp)
target_link_libraries(tmgm_ensemble_predict PRIVATE
    tmgm_dataset tmgm_ensemble)
if(MSVC)
    target_compile_options(tmgm_ensemble_predict PRIVATE /W4 /permissive-)
else()
    target_compile_options(tmgm_ensemble_predict PRIVATE
        -Wall -Wextra -Wpedantic)
endif()

if(TMGM_HAS_CUDA)
    add_executable(tmgm_train src/train_main.cpp)
    target_link_libraries(tmgm_train PRIVATE tmgm_dataset tmgm_cuda)
    set_property(TARGET tmgm_train PROPERTY CUDA_RUNTIME_LIBRARY Shared)
    if(MSVC)
        target_compile_options(tmgm_train PRIVATE /W4 /permissive-)
    else()
        target_compile_options(tmgm_train PRIVATE -Wall -Wextra -Wpedantic)
    endif()

    # Deliberately separate from the production trainer. This executable is
    # byte-identical in its default mode, but exposes the opt-in output-order
    # ablation CLI so experiments cannot enable it accidentally.
    add_executable(tmgm_train_order_ablation src/train_main.cpp)
    target_compile_definitions(tmgm_train_order_ablation PRIVATE
        TMGM_ENABLE_OUTPUT_ORDER_ABLATION=1)
    target_link_libraries(tmgm_train_order_ablation PRIVATE
        tmgm_dataset tmgm_cuda)
    set_property(TARGET tmgm_train_order_ablation
        PROPERTY CUDA_RUNTIME_LIBRARY Shared)
    if(MSVC)
        target_compile_options(tmgm_train_order_ablation PRIVATE
            /W4 /permissive-)
    else()
        target_compile_options(tmgm_train_order_ablation PRIVATE
            -Wall -Wextra -Wpedantic)
    endif()

    add_executable(tmgm_predict src/predict_main.cpp)
    target_link_libraries(tmgm_predict PRIVATE
        tmgm_dataset tmgm_model tmgm_tm_calibration tmgm_cuda)
    set_property(TARGET tmgm_predict PROPERTY CUDA_RUNTIME_LIBRARY Shared)
    if(MSVC)
        target_compile_options(tmgm_predict PRIVATE /W4 /permissive-)
    else()
        target_compile_options(tmgm_predict PRIVATE -Wall -Wextra -Wpedantic)
    endif()
endif()
if(MSVC)
    target_compile_options(tmgm_dataset_info PRIVATE /W4 /permissive-)
else()
    target_compile_options(tmgm_dataset_info PRIVATE -Wall -Wextra -Wpedantic)
endif()

include(CTest)
if(BUILD_TESTING)
    add_executable(tmgm_causal_resampler_tests
        tests/causal_resampler_test.cpp)
    target_link_libraries(tmgm_causal_resampler_tests PRIVATE tmgm_resampler)
    if(MSVC)
        target_compile_options(tmgm_causal_resampler_tests PRIVATE
            /W4 /permissive- /fp:strict)
    else()
        target_compile_options(tmgm_causal_resampler_tests PRIVATE
            -Wall -Wextra -Wpedantic -ffp-contract=off)
    endif()
    add_test(NAME causal_44100_48000_to_22050_resampler
        COMMAND tmgm_causal_resampler_tests)

    add_executable(tmgm_dataset_tests tests/dataset_loader_test.cpp)
    target_link_libraries(tmgm_dataset_tests PRIVATE tmgm_dataset)
    add_test(NAME dataset_roundtrip_and_validation COMMAND tmgm_dataset_tests)

    add_executable(tmgm_tm_reference_tests tests/tm_reference_test.cpp)
    target_link_libraries(tmgm_tm_reference_tests PRIVATE tmgm_tm_reference)
    add_test(NAME tm_shared_clause_cpu_reference COMMAND tmgm_tm_reference_tests)

    add_executable(tmgm_model_tests tests/model_test.cpp)
    target_link_libraries(tmgm_model_tests PRIVATE tmgm_model)
    add_test(NAME tm_native_model_roundtrip COMMAND tmgm_model_tests)

    add_executable(tmgm_tm_calibration_tests tests/tm_calibration_test.cpp)
    target_link_libraries(tmgm_tm_calibration_tests PRIVATE tmgm_tm_calibration)
    add_test(NAME tm_validation_threshold_calibration COMMAND tmgm_tm_calibration_tests)

    add_executable(tmgm_tm_validation_tests tests/tm_validation_test.cpp)
    target_link_libraries(tmgm_tm_validation_tests PRIVATE tmgm_tm_validation)
    add_test(NAME tm_held_out_validation_support COMMAND tmgm_tm_validation_tests)

    add_executable(tmgm_tm_output_order_tests
        tests/tm_output_order_test.cpp)
    target_link_libraries(tmgm_tm_output_order_tests PRIVATE
        tmgm_model)
    if(MSVC)
        target_compile_options(tmgm_tm_output_order_tests PRIVATE
            /W4 /permissive-)
    else()
        target_compile_options(tmgm_tm_output_order_tests PRIVATE
            -Wall -Wextra -Wpedantic)
    endif()
    add_test(NAME tm_output_update_order_ablation
        COMMAND tmgm_tm_output_order_tests)

    add_executable(tmgm_ensemble_inference_tests
        tests/ensemble_inference_test.cpp)
    target_link_libraries(tmgm_ensemble_inference_tests PRIVATE
        tmgm_dataset tmgm_ensemble)
    if(MSVC)
        target_compile_options(tmgm_ensemble_inference_tests PRIVATE
            /W4 /permissive-)
    else()
        target_compile_options(tmgm_ensemble_inference_tests PRIVATE
            -Wall -Wextra -Wpedantic)
    endif()
    add_test(NAME tm_ensemble_sparse_cpu_inference
        COMMAND tmgm_ensemble_inference_tests)

    set(TMGM_PRODUCTION_ENSEMBLE_FIXTURE
        "${CMAKE_CURRENT_SOURCE_DIR}/../artifacts/native-full-natural-d2w3/production-ensemble/onset-qgrid-v1/production.tmgmbundle")
    if(EXISTS "${TMGM_PRODUCTION_ENSEMBLE_FIXTURE}")
        add_executable(tmgm_ensemble_bundle_tests
            tests/ensemble_bundle_test.cpp)
        target_link_libraries(tmgm_ensemble_bundle_tests PRIVATE tmgm_ensemble)
        if(MSVC)
            target_compile_options(tmgm_ensemble_bundle_tests PRIVATE
                /W4 /permissive-)
        else()
            target_compile_options(tmgm_ensemble_bundle_tests PRIVATE
                -Wall -Wextra -Wpedantic)
        endif()
        add_test(NAME tm_ensemble_production_bundle_loader
            COMMAND tmgm_ensemble_bundle_tests
                "${TMGM_PRODUCTION_ENSEMBLE_FIXTURE}")

        set(TMGM_PRODUCTION_PARITY_DATASET
            "${CMAKE_CURRENT_SOURCE_DIR}/../artifacts/native-full-natural-d2w3/production-ensemble/onset-qgrid-v1/parity-128.tmgd")
        set(TMGM_PRODUCTION_ACTIVITY_ORACLE
            "${CMAKE_CURRENT_SOURCE_DIR}/../artifacts/native-full-natural-d2w3/production-ensemble/onset-qgrid-v1/parity-cpu/activity-python-v2.tsv")
        if(EXISTS "${TMGM_PRODUCTION_PARITY_DATASET}" AND
           EXISTS "${TMGM_PRODUCTION_ACTIVITY_ORACLE}")
            add_executable(tmgm_ensemble_production_parity_tests
                tests/ensemble_production_parity_test.cpp)
            target_link_libraries(tmgm_ensemble_production_parity_tests PRIVATE
                tmgm_dataset tmgm_ensemble)
            if(MSVC)
                target_compile_options(tmgm_ensemble_production_parity_tests PRIVATE
                    /W4 /permissive-)
            else()
                target_compile_options(tmgm_ensemble_production_parity_tests PRIVATE
                    -Wall -Wextra -Wpedantic)
            endif()
            add_test(NAME tm_ensemble_production_raw_and_fused_parity
                COMMAND tmgm_ensemble_production_parity_tests
                    "${CMAKE_CURRENT_SOURCE_DIR}/..")
        endif()
    endif()

    set(TMGM_STRICT_CAP16_V3_FIXTURE
        "${CMAKE_CURRENT_SOURCE_DIR}/../artifacts/native-listening-comparison-20260718-strict-cap16-v3")
    if(EXISTS "${TMGM_STRICT_CAP16_V3_FIXTURE}/bundles/plain.tmgmbundle" AND
       EXISTS "${TMGM_STRICT_CAP16_V3_FIXTURE}/features/plain/2222.tmgd" AND
       EXISTS "${TMGM_STRICT_CAP16_V3_FIXTURE}/scores/2222/activity-final.tsv" AND
       EXISTS "${TMGM_STRICT_CAP16_V3_FIXTURE}/scores/2222/onset-primary.tsv")
        add_executable(tmgm_strict_cap16_v3_parity_tests
            tests/strict_cap16_v3_parity_test.cpp)
        target_link_libraries(tmgm_strict_cap16_v3_parity_tests PRIVATE
            tmgm_dataset tmgm_ensemble)
        if(MSVC)
            target_compile_options(tmgm_strict_cap16_v3_parity_tests PRIVATE
                /W4 /permissive- /fp:strict)
        else()
            target_compile_options(tmgm_strict_cap16_v3_parity_tests PRIVATE
                -Wall -Wextra -Wpedantic -ffp-contract=off)
        endif()
        add_test(NAME tm_strict_cap16_v3_cross_bank_realtime_parity
            COMMAND tmgm_strict_cap16_v3_parity_tests
                "${TMGM_STRICT_CAP16_V3_FIXTURE}")
    endif()

    if(EXISTS "${TMGM_STRICT_CAP16_V3_FIXTURE}/native-frontend/strict-cap16-v3.tmgmfront" AND
       EXISTS "${TMGM_STRICT_CAP16_V3_FIXTURE}/native-frontend/2222-mono-22050.f32le" AND
       EXISTS "${TMGM_STRICT_CAP16_V3_FIXTURE}/features/plain/2222.tmgd" AND
       EXISTS "${TMGM_STRICT_CAP16_V3_FIXTURE}/features/hcontrast-d2/2222.tmgd" AND
       EXISTS "${TMGM_STRICT_CAP16_V3_FIXTURE}/features/hprofile-d3/2222.tmgd" AND
       EXISTS "${TMGM_STRICT_CAP16_V3_FIXTURE}/features/cattack-d3/2222.tmgd")
        add_executable(tmgm_strict_cap16_v3_frontend_parity_tests
            tests/strict_cap16_v3_frontend_parity_test.cpp)
        target_link_libraries(tmgm_strict_cap16_v3_frontend_parity_tests PRIVATE
            tmgm_dataset tmgm_frontend)
        if(MSVC)
            target_compile_options(
                tmgm_strict_cap16_v3_frontend_parity_tests PRIVATE
                /W4 /permissive- /fp:strict)
        else()
            target_compile_options(
                tmgm_strict_cap16_v3_frontend_parity_tests PRIVATE
                -Wall -Wextra -Wpedantic -ffp-contract=off)
        endif()
        add_test(NAME tm_strict_cap16_v3_streaming_frontend_parity
            COMMAND tmgm_strict_cap16_v3_frontend_parity_tests
                "${TMGM_STRICT_CAP16_V3_FIXTURE}")
    endif()

    if(TMGM_HAS_CUDA)
        add_executable(tmgm_cuda_inference_parity_tests
            tests/cuda_inference_parity_test.cpp)
        target_link_libraries(tmgm_cuda_inference_parity_tests PRIVATE
            tmgm_dataset tmgm_model tmgm_tm_reference tmgm_cuda)
        set_property(TARGET tmgm_cuda_inference_parity_tests
            PROPERTY CUDA_RUNTIME_LIBRARY Shared)
        if(MSVC)
            target_compile_options(tmgm_cuda_inference_parity_tests PRIVATE
                /W4 /permissive-)
        else()
            target_compile_options(tmgm_cuda_inference_parity_tests PRIVATE
                -Wall -Wextra -Wpedantic)
        endif()
        add_test(NAME tm_cuda_cpu_inference_parity
            COMMAND tmgm_cuda_inference_parity_tests)
        set_tests_properties(tm_cuda_cpu_inference_parity PROPERTIES
            SKIP_RETURN_CODE 77)
    endif()
endif()

message(STATUS "TMGM native CUDA backend: ${TMGM_HAS_CUDA}")
