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

add_library(plaidml SHARED IMPORTED)

set_target_properties(plaidml PROPERTIES IMPORTED_LOCATION ${PLAIDML_SO_PATH})

ie_add_plugin(NAME ${TARGET_NAME}
              DEVICE_NAME "PLAIDML"
              OBJECT_LIBRARIES ${object_libraries}
              SOURCES ${SOURCES} ${HEADERS})

set(ngraph_DIR ${CMAKE_BINARY_DIR}/ngraph)
find_package(ngraph REQUIRED)

target_link_libraries(${TARGET_NAME} PRIVATE inference_engine ${NGRAPH_LIBRARIES} plaidml)

include_directories(
        ${PLAIDML_SRC_PATH}
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_BINARY_DIR}/include)
