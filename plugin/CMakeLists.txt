set(TMGM_PLUGIN_PRODUCT_NAME "TM Guitar MIDI")

set(_tmgm_formats)
if(TMGM_BUILD_VST3)
    list(APPEND _tmgm_formats VST3)
endif()
if(TMGM_BUILD_LV2)
    list(APPEND _tmgm_formats LV2)
endif()

add_library(tmgm_realtime_engine STATIC
    TmRealtimeEngine.cpp
    TmRealtimeEngine.hpp
    CleanroomAcousticFrontend.cpp
    CleanroomAcousticFrontend.hpp
)
target_include_directories(tmgm_realtime_engine
    PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}"
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../native/third_party/pocketfft")
target_link_libraries(tmgm_realtime_engine
    PUBLIC tmgm_frontend tmgm_ensemble tmgm_resampler)
target_compile_features(tmgm_realtime_engine PUBLIC cxx_std_20)
if(MSVC)
    target_compile_options(tmgm_realtime_engine PRIVATE
        /W4 /permissive- /EHsc /fp:strict)
else()
    target_compile_options(tmgm_realtime_engine PRIVATE
        -Wall -Wextra -Wpedantic -ffp-contract=off)
endif()

juce_add_plugin(TM_GuitarMidi
    VERSION "${TM_Guitar_Midi_VERSION}"
    COMPANY_NAME "TM Guitar MIDI"
    BUNDLE_ID "org.tmguitarmidi.plugin"
    PLUGIN_MANUFACTURER_CODE TMgm
    PLUGIN_CODE Tmgu
    FORMATS ${_tmgm_formats}
    PRODUCT_NAME "${TMGM_PLUGIN_PRODUCT_NAME}"
    DESCRIPTION "Realtime polyphonic guitar-to-MIDI using Tsetlin Machines"
    LV2URI "urn:tmgm:plugin:tm-guitar-midi"
    IS_SYNTH FALSE
    NEEDS_MIDI_INPUT FALSE
    NEEDS_MIDI_OUTPUT TRUE
    IS_MIDI_EFFECT FALSE
    EDITOR_WANTS_KEYBOARD_FOCUS FALSE
    COPY_PLUGIN_AFTER_BUILD FALSE
)
unset(_tmgm_formats)

target_sources(TM_GuitarMidi PRIVATE
    PluginProcessor.cpp
    PluginProcessor.hpp
)
target_link_libraries(TM_GuitarMidi
    PRIVATE
        tmgm_realtime_engine
        juce::juce_audio_processors
        ${CMAKE_DL_LIBS}
    PUBLIC
        juce::juce_recommended_config_flags
        juce::juce_recommended_lto_flags
        juce::juce_recommended_warning_flags
)
target_compile_definitions(TM_GuitarMidi PUBLIC
    JUCE_WEB_BROWSER=0
    JUCE_USE_CURL=0
    JUCE_VST3_CAN_REPLACE_VST2=0
)
juce_generate_juce_header(TM_GuitarMidi)

function(tmgm_attach_model_package target format)
    if(format STREQUAL "VST3")
        # All JUCE VST3 bundles use Contents/<architecture> for the module and
        # Contents/Resources for data on Windows, macOS, and Linux.
        set(_destination
            "$<TARGET_FILE_DIR:${target}>/../Resources/TMModel")
    elseif(format STREQUAL "LV2")
        # JUCE places the shared object and TTL manifests directly in .lv2.
        set(_destination "$<TARGET_FILE_DIR:${target}>/TMModel")
    else()
        message(FATAL_ERROR "Unsupported model resource format: ${format}")
    endif()

    add_custom_command(TARGET ${target} POST_BUILD
        COMMAND "${CMAKE_COMMAND}"
            "-DTMGM_MODEL_SOURCE=${TMGM_MODEL_PACKAGE}"
            "-DTMGM_MODEL_DESTINATION=${_destination}"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/../cmake/CopyModelPackage.cmake"
        COMMENT "Authenticating and packaging strict-cap16-v3 model resources"
        VERBATIM
    )

    # JUCE's ad-hoc signing step runs before user POST_BUILD commands. Adding
    # resources afterwards invalidates a macOS bundle signature, so sign the
    # completed bundle once more. A release workflow may replace this ad-hoc
    # signature with Developer ID signing and notarization later.
    if(APPLE AND format STREQUAL "VST3")
        add_custom_command(TARGET ${target} POST_BUILD
            COMMAND /usr/bin/codesign --force --deep --sign -
                "$<TARGET_BUNDLE_DIR:${target}>"
            COMMENT "Ad-hoc signing the complete VST3 bundle"
            VERBATIM
        )
    endif()
endfunction()

if(TMGM_BUILD_VST3)
    tmgm_attach_model_package(TM_GuitarMidi_VST3 VST3)
endif()
if(TMGM_BUILD_LV2)
    tmgm_attach_model_package(TM_GuitarMidi_LV2 LV2)
endif()

if(TMGM_BUILD_TESTS)
    add_executable(tmgm_engine_smoke EngineSmoke.cpp)
    target_link_libraries(tmgm_engine_smoke PRIVATE tmgm_realtime_engine)
    target_compile_features(tmgm_engine_smoke PRIVATE cxx_std_20)

    add_executable(tmgm_cleanroom_frontend_parity
        CleanroomAcousticFrontendParityTest.cpp)
    target_link_libraries(tmgm_cleanroom_frontend_parity
        PRIVATE tmgm_realtime_engine)
    target_compile_features(tmgm_cleanroom_frontend_parity PRIVATE cxx_std_20)
    add_test(NAME cleanroom_acoustic_frontend_parity
        COMMAND tmgm_cleanroom_frontend_parity
            "${CMAKE_CURRENT_SOURCE_DIR}/CleanroomAcousticFrontendFixture.tsv")

    function(tmgm_add_host_smoke name wrapper format)
        juce_add_console_app(${name}
            PRODUCT_NAME "TM Guitar MIDI ${format} host smoke")
        target_sources(${name} PRIVATE HostSmoke.cpp)
        target_link_libraries(${name} PRIVATE
            juce::juce_audio_processors
            juce::juce_audio_formats)
        target_compile_definitions(${name} PRIVATE
            JUCE_WEB_BROWSER=0
            JUCE_USE_CURL=0
            JUCE_PLUGINHOST_VST=0
            JUCE_PLUGINHOST_LADSPA=0)
        if(format STREQUAL "VST3")
            target_compile_definitions(${name} PRIVATE
                TMGM_HOST_TEST_VST3=1
                JUCE_PLUGINHOST_VST3=1
                JUCE_PLUGINHOST_LV2=0)
        elseif(format STREQUAL "LV2")
            target_compile_definitions(${name} PRIVATE
                TMGM_HOST_TEST_LV2=1
                JUCE_PLUGINHOST_VST3=0
                JUCE_PLUGINHOST_LV2=1)
        endif()
        target_compile_features(${name} PRIVATE cxx_std_20)
        add_dependencies(${name} ${wrapper})
        if(MSVC)
            target_compile_options(${name} PRIVATE /W4 /permissive- /EHsc)
        else()
            target_compile_options(${name} PRIVATE
                -Wall -Wextra -Wpedantic)
        endif()
        get_target_property(_artefact ${wrapper} JUCE_PLUGIN_ARTEFACT_FILE)
        add_test(NAME ${name}
            COMMAND ${name} "${_artefact}")
        unset(_artefact)
    endfunction()

    if(TMGM_BUILD_VST3)
        tmgm_add_host_smoke(tmgm_vst3_host_smoke
            TM_GuitarMidi_VST3 VST3)
    endif()
    if(TMGM_BUILD_LV2)
        tmgm_add_host_smoke(tmgm_lv2_host_smoke
            TM_GuitarMidi_LV2 LV2)
    endif()
endif()
