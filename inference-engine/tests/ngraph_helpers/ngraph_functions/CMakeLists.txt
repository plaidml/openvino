# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME ngraphFunctions)

list(APPEND EXPORT_DEPENDENCIES
        ${NGRAPH_LIBRARIES}
        ngraph_backend
        interpreter_backend
        )

set(PUBLIC_HEADERS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")

addIeTarget(
        NAME ${TARGET_NAME}
        TYPE STATIC
        ROOT ${PUBLIC_HEADERS_DIR}
        INCLUDES
            PUBLIC
                ${PUBLIC_HEADERS_DIR}
        ADDITIONAL_SOURCE_DIRS
            ${CMAKE_CURRENT_SOURCE_DIR}/src
        LINK_LIBRARIES
            PUBLIC        
                ${NGRAPH_LIBRARIES}
                ngraph_backend
                interpreter_backend
        ADD_CPPLINT
        DEVELOPER_PACKAGE
            inference_engine_tests
)

ie_faster_build(${TARGET_NAME}
    UNITY
    PCH PRIVATE "src/precomp.hpp"
)
