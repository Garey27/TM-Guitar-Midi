if(NOT DEFINED TMGM_MODEL_SOURCE OR NOT DEFINED TMGM_MODEL_DESTINATION)
    message(FATAL_ERROR
        "TMGM_MODEL_SOURCE and TMGM_MODEL_DESTINATION are required")
endif()

include("${CMAKE_CURRENT_LIST_DIR}/VerifyModelPackage.cmake")
tmgm_verify_model_package("${TMGM_MODEL_SOURCE}")

foreach(_relative IN LISTS TMGM_MODEL_RELATIVE_FILES)
    get_filename_component(_directory "${_relative}" DIRECTORY)
    file(MAKE_DIRECTORY "${TMGM_MODEL_DESTINATION}/${_directory}")
    file(COPY_FILE
        "${TMGM_MODEL_SOURCE}/${_relative}"
        "${TMGM_MODEL_DESTINATION}/${_relative}"
        ONLY_IF_DIFFERENT)
endforeach()

tmgm_verify_model_package("${TMGM_MODEL_DESTINATION}")
