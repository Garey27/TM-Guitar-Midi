# The runtime authenticates these files again while loading them. This CMake
# check catches incomplete source trees and packaging mistakes earlier.
set(TMGM_MODEL_RELATIVE_FILES
    "bundles/plain.tmgmbundle"
    "bundles/hcontrast-d2.tmgmbundle"
    "bundles/hcontrast-d3.tmgmbundle"
    "bundles/hprofile-d3.tmgmbundle"
    "bundles/cattack-d3.tmgmbundle"
    "native-frontend/strict-cap16-v3.tmgmfront"
)

set(_TMGM_MODEL_SHA256
    "0e2e3cc4452da523839e2b78fc43db7b47db5141aee7c46830057d3c4df4d5b2"
    "f4dc0c15f0f7ae68527c6dfabd927bf5c0649f99b9479a99da5fc1c8c6c91ad2"
    "a2fbdc15118ad4d76c4bd7dc9038fb67975d889097fd28c14eef2ecead9a34c2"
    "8164880768ba265614bde41fa85d071a1ca8ae9648ac553ba82daa7081e922b0"
    "69ab06035dc6f8fc2cdd6de83b71a5e8e2cf81881ff111a47ca9ea55648ab1f0"
    "953289a46e242c8d61181d80ad01c9d9dffb6a6f82f11e74ad8b76ad02a1f0bf"
)

function(tmgm_verify_model_package root)
    list(LENGTH TMGM_MODEL_RELATIVE_FILES _file_count)
    list(LENGTH _TMGM_MODEL_SHA256 _hash_count)
    if(NOT _file_count EQUAL _hash_count)
        message(FATAL_ERROR "Internal model checksum table is inconsistent")
    endif()

    math(EXPR _last "${_file_count} - 1")
    foreach(_index RANGE 0 ${_last})
        list(GET TMGM_MODEL_RELATIVE_FILES ${_index} _relative)
        list(GET _TMGM_MODEL_SHA256 ${_index} _expected)
        set(_path "${root}/${_relative}")
        if(NOT EXISTS "${_path}")
            message(FATAL_ERROR "Missing model resource: ${_path}")
        endif()
        file(SHA256 "${_path}" _actual)
        string(TOLOWER "${_actual}" _actual)
        if(NOT _actual STREQUAL _expected)
            message(FATAL_ERROR
                "Model checksum mismatch for ${_path}: expected ${_expected}, got ${_actual}")
        endif()
    endforeach()
endfunction()

# Script mode used by CI to authenticate resources inside a built bundle.
if(DEFINED TMGM_VERIFY_MODEL_ROOT)
    tmgm_verify_model_package("${TMGM_VERIFY_MODEL_ROOT}")
    message(STATUS "Authenticated model package: ${TMGM_VERIFY_MODEL_ROOT}")
endif()
