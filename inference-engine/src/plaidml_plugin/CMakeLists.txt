# Copyright (C) 2018-2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "PlaidMLPlugin")

file(GLOB SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/ops/*.cpp
        ${OS_SPECIFIC_SRC}
)

file(GLOB HEADERS
        ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp
)

addVersionDefines(plaidml_plugin.cpp CI_BUILD_NUMBER PLAIDML_VERSION)

include_directories(
        $<TARGET_PROPERTY:inference_engine_plugin_api,INTERFACE_INCLUDE_DIRECTORIES>
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_BINARY_DIR}/include)

ie_add_plugin(NAME ${TARGET_NAME}
              DEVICE_NAME "PLAIDML"
              OBJECT_LIBRARIES ${object_libraries}
              SOURCES ${SOURCES} ${HEADERS})

target_link_libraries(${TARGET_NAME} PRIVATE inference_engine inference_engine_lp_transformations
                      inference_engine_transformations
                      ${INTEL_ITT_LIBS} plaidml)




add_library(${TARGET_NAME}_obj OBJECT ${SOURCES} ${HEADERS})


