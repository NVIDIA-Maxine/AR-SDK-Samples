# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

#################
# ExpressionApp #
#################

set(REQUIRED_FEATURES nvARFaceExpressions:1.1.0.0 nvARLandmarkDetection:1.1.0.0 nvARFaceBoxDetection:1.1.0.0)
foreach(REQUIRED_FEATURE_AND_VERSION IN LISTS REQUIRED_FEATURES)
  verify_app_feature_dependency(${REQUIRED_FEATURE_AND_VERSION} RESULT)
  if(NOT RESULT)
    message(${SAMPLES_DEPS_ERROR_LEVEL} "Skipping ExpressionApp.")
    return()
  endif()
endforeach()

option(ENABLE_UI "Enable UI for Expression App" OFF)
if(${ENABLE_UI})
  add_definitions("-D_ENABLE_UI")
endif()

if(WIN32)
  set(RUN_FILES
    run_expressionapp_offline.bat
    run_expressionapp_webcam.bat
  )
else()
  set(RUN_FILES
    run_expressionapp_offline.sh
    run_expressionapp_webcam.sh
    setup_env.sh)
endif()

set(APP_CORE_SRCS
  ExpressionApp.cpp
  meshRenderer.cpp meshRenderer.h
  directoryIterator.cpp directoryIterator.h
)

set(GL_BACKEND_SRCS
  ${CMAKE_CURRENT_SOURCE_DIR}/backendopengl/glMaterial.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/backendopengl/glMaterial.h
  ${CMAKE_CURRENT_SOURCE_DIR}/backendopengl/glMesh.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/backendopengl/glMesh.h
  ${CMAKE_CURRENT_SOURCE_DIR}/backendopengl/glShaders.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/backendopengl/glShaders.h
  ${CMAKE_CURRENT_SOURCE_DIR}/backendopengl/glSpectrum.h
  ${CMAKE_CURRENT_SOURCE_DIR}/backendopengl/simpleFaceModel.h
  ${CMAKE_CURRENT_SOURCE_DIR}/backendopengl/openGLMeshRenderer.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/backendopengl/openGLMeshRenderer.h
  ${CMAKE_CURRENT_SOURCE_DIR}/backendopengl/faceIO.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/backendopengl/faceIO.h
)
set(APP_SRCS ${APP_CORE_SRCS} ${GL_BACKEND_SRCS})

set(UI_SRCS
  expressionAppUI.h
  expressionAppUI.cpp
)
set(IMGUI_SRCS
  ${CMAKE_CURRENT_SOURCE_DIR}/../../external/Imgui/imgui-docking/imgui.h
  ${CMAKE_CURRENT_SOURCE_DIR}/../../external/Imgui/imgui-docking/imconfig.h
  ${CMAKE_CURRENT_SOURCE_DIR}/../../external/Imgui/imgui-docking/imgui_internal.h
  ${CMAKE_CURRENT_SOURCE_DIR}/../../external/Imgui/imgui-docking/imstb_textedit.h
  ${CMAKE_CURRENT_SOURCE_DIR}/../../external/Imgui/imgui-docking/imstb_rectpack.h
  ${CMAKE_CURRENT_SOURCE_DIR}/../../external/Imgui/imgui-docking/imstb_truetype.h
  ${CMAKE_CURRENT_SOURCE_DIR}/../../external/Imgui/imgui-docking/backends/imgui_impl_glfw.h
  ${CMAKE_CURRENT_SOURCE_DIR}/../../external/Imgui/imgui-docking/backends/imgui_impl_opengl3.h
  ${CMAKE_CURRENT_SOURCE_DIR}/../../external/Imgui/imgui-docking/backends/imgui_impl_opengl3_loader.h
  ${CMAKE_CURRENT_SOURCE_DIR}/../../external/Imgui/imgui-docking/misc/cpp/imgui_stdlib.h
  ${CMAKE_CURRENT_SOURCE_DIR}/../../external/Imgui/imgui-docking/imgui_tables.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../../external/Imgui/imgui-docking/imgui_widgets.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../../external/Imgui/imgui-docking/imgui.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../../external/Imgui/imgui-docking/imgui_draw.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../../external/Imgui/imgui-docking/backends/imgui_impl_glfw.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../../external/Imgui/imgui-docking/backends/imgui_impl_opengl3.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../../external/Imgui/imgui-docking/misc/cpp/imgui_stdlib.cpp
)

if(${ENABLE_UI})
  set(APP_SRCS ${APP_SRCS} ${UI_SRCS} ${IMGUI_SRCS})
endif(${ENABLE_UI})

add_executable(ExpressionApp README.md ${APP_SRCS})

if(${ENABLE_UI})
  target_include_directories(ExpressionApp PRIVATE
    /../../external/Imgui/imgui-docking
    /../../external/nlohmann/json/single_include/nlohmann
  )
endif()

if(TARGET OpenCV)
  set(OPENCV OpenCV)
else()
  set(OPENCV opencv346)
endif()

target_include_directories(ExpressionApp PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/backendopengl
  ${ARSDKSampleApps_utils_DIR}
  ${OpenCV_INCLUDE_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR}/../../external/nlohmann/json/single_include
  ${CMAKE_CURRENT_SOURCE_DIR}/../../external/Imgui/imgui-docking
)

target_link_libraries(ExpressionApp PRIVATE
  nvARPose
  NVCVImage
  GLM
  ${OPENCV}
)

if(WIN32)
  target_link_directories(ExpressionApp PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../../external/GLAD/lib
    ${CMAKE_CURRENT_SOURCE_DIR}/../../external/GLFW/lib
  )
  target_include_directories(ExpressionApp PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../external/GLAD/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../external/GLFW/include
  )
  target_link_libraries(ExpressionApp PRIVATE
    GLAD
    ${OPENGL_gl_LIBRARY}
    glfw3
  )
else()
  target_link_libraries(ExpressionApp PRIVATE
    OpenGL
    glfw
    dl
  )
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fpermissive")
endif()

get_target_property(NVAR_DYNAMIC_LIBRARY_DIR nvARPose DYNAMIC_LIBRARY_DIR)
set(NVAR_DYNAMIC_LIBRARY_DIRS ${NVAR_DYNAMIC_LIBRARY_DIR})

foreach(REQUIRED_FEATURE_AND_VERSION IN LISTS REQUIRED_FEATURES)
  string(REGEX MATCH "^([^:]+):([^:]+)" _ "${REQUIRED_FEATURE_AND_VERSION}")
  set(REQUIRED_FEATURE_NAME ${CMAKE_MATCH_1})
  target_link_libraries(ExpressionApp PRIVATE ${REQUIRED_FEATURE_NAME})
  get_target_property(FEATURE_DYNAMIC_LIBRARY_DIR ${REQUIRED_FEATURE_NAME} INTERFACE_DYNAMIC_LIBRARY_DIRECTORY)
  if(FEATURE_DYNAMIC_LIBRARY_DIR)
    list(APPEND NVAR_DYNAMIC_LIBRARY_DIRS ${FEATURE_DYNAMIC_LIBRARY_DIR})
  endif()
endforeach()

get_target_property(NVAR_MODEL_DIR nvARPose MODEL_DIR)

if(WIN32)
  get_target_property(OPENCV_PATH_STR ${OPENCV} DYNAMIC_LIBRARY_DIR)
  set(PATH_STR "PATH=%PATH%" ${OPENCV_PATH_STR};${NVAR_DYNAMIC_LIBRARY_DIRS})
  set(CMD_ARG_STR "--model_path=\"${NVAR_MODEL_DIR}\"")
  set_target_properties(ExpressionApp PROPERTIES
    FOLDER SampleApps
    VS_DEBUGGER_ENVIRONMENT "${PATH_STR}"
    VS_DEBUGGER_COMMAND_ARGUMENTS "${CMD_ARG_STR}"
  )

  # https://stackoverflow.com/questions/28178978/how-to-generate-pdb-files-for-release-build-with-cmake-flags
  target_compile_options(ExpressionApp PRIVATE $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CXX>>:/Zi>) # Keep flag specific to C++ to avoid Cuda build errors
  target_link_options(ExpressionApp PRIVATE $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CXX>>:/INCREMENTAL:NO;/DEBUG;/OPT:REF;/OPT:ICF>)
endif()

# Configure run-files
foreach(RUN_FILE IN LISTS RUN_FILES)
  # Note: Separate configure_file from file copy for older CMake version to support file permissions
  configure_file(${RUN_FILE}.in ${CMAKE_CURRENT_BINARY_DIR}/tmp/${RUN_FILE} @ONLY)
  file(COPY ${CMAKE_CURRENT_BINARY_DIR}/tmp/${RUN_FILE} DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/run_files FILE_PERMISSIONS OWNER_EXECUTE OWNER_READ OWNER_WRITE)
  file(REMOVE_RECURSE ${CMAKE_CURRENT_BINARY_DIR}/tmp/)
  add_custom_command(TARGET ExpressionApp POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_CURRENT_BINARY_DIR}/run_files/${RUN_FILE}
    $<TARGET_FILE_DIR:ExpressionApp>)
endforeach()

# Add resources and README.md file
add_custom_command(TARGET ExpressionApp POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
  ${ARSDKSampleApps_RESOURCES_DIR}/faceTrack.mp4
  ${CMAKE_CURRENT_SOURCE_DIR}/README.md
  $<TARGET_FILE_DIR:ExpressionApp>
)
