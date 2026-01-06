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

################
# BodyTrackApp #
################

set(REQUIRED_FEATURES nvARBodyPoseEstimation:1.1.0.0 nvARBodyDetection:1.1.0.0)
foreach(REQUIRED_FEATURE_AND_VERSION IN LISTS REQUIRED_FEATURES)
  verify_app_feature_dependency(${REQUIRED_FEATURE_AND_VERSION} RESULT)
  if(NOT RESULT)
    message(${SAMPLES_DEPS_ERROR_LEVEL} "Skipping BodyTrackApp.")
    return()
  endif()
endforeach()

if(WIN32)
  set(RUN_FILES
    run_bodytrackapp_offline.bat
    run_bodytrackapp_webcam.bat
  )
else()
  set(RUN_FILES
    run_bodytrackapp_offline.sh
    run_bodytrackapp_webcam.sh
    setup_env.sh)
endif()

set(APP_SRCS
  BodyTrackApp.cpp
  bodyEngine.cpp
  bodyEngine.h
  ${ARSDKSampleApps_utils_DIR}/renderingUtils.cpp
  ${ARSDKSampleApps_utils_DIR}/renderingUtils.h
)

add_executable(BodyTrackApp README.md ${APP_SRCS})

if(TARGET OpenCV)
  set(OPENCV OpenCV)
else()
  set(OPENCV opencv346)
endif()

target_link_libraries(BodyTrackApp PRIVATE
  GLM
  ${OPENCV}
  ${CMAKE_DL_LIBS}
  nvARPose
  NVCVImage
)

target_include_directories(BodyTrackApp PRIVATE
  ${ARSDKSampleApps_utils_DIR}
  ${OpenCV_INCLUDE_DIRS}
)

get_target_property(NVAR_DYNAMIC_LIBRARY_DIR nvARPose DYNAMIC_LIBRARY_DIR)
set(NVAR_DYNAMIC_LIBRARY_DIRS ${NVAR_DYNAMIC_LIBRARY_DIR})

foreach(REQUIRED_FEATURE_AND_VERSION IN LISTS REQUIRED_FEATURES)
  string(REGEX MATCH "^([^:]+):([^:]+)" _ "${REQUIRED_FEATURE_AND_VERSION}")
  set(REQUIRED_FEATURE_NAME ${CMAKE_MATCH_1})
  target_link_libraries(BodyTrackApp PRIVATE ${REQUIRED_FEATURE_NAME})
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
  set_target_properties(BodyTrackApp PROPERTIES
    FOLDER SampleApps
    VS_DEBUGGER_ENVIRONMENT "${PATH_STR}"
    VS_DEBUGGER_COMMAND_ARGUMENTS "${CMD_ARG_STR}"
  )

  # https://stackoverflow.com/questions/28178978/how-to-generate-pdb-files-for-release-build-with-cmake-flags
  target_compile_options(BodyTrackApp PRIVATE $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CXX>>:/Zi>) # Keep flag specific to C++ to avoid Cuda build errors
  target_link_options(BodyTrackApp PRIVATE $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CXX>>:/INCREMENTAL:NO;/DEBUG;/OPT:REF;/OPT:ICF>)
endif()

# Configure run-files
foreach(RUN_FILE IN LISTS RUN_FILES)
  # Note: Separate configure_file from file copy for older CMake version to support file permissions
  configure_file(${RUN_FILE}.in ${CMAKE_CURRENT_BINARY_DIR}/tmp/${RUN_FILE} @ONLY)
  file(COPY ${CMAKE_CURRENT_BINARY_DIR}/tmp/${RUN_FILE} DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/run_files FILE_PERMISSIONS OWNER_EXECUTE OWNER_READ OWNER_WRITE)
  file(REMOVE_RECURSE ${CMAKE_CURRENT_BINARY_DIR}/tmp/)
  add_custom_command(TARGET BodyTrackApp POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_CURRENT_BINARY_DIR}/run_files/${RUN_FILE}
    $<TARGET_FILE_DIR:BodyTrackApp>)
endforeach()

# Add resources and README.md file
add_custom_command(TARGET BodyTrackApp POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
  ${ARSDKSampleApps_RESOURCES_DIR}/bodyTrack.mp4
  ${CMAKE_CURRENT_SOURCE_DIR}/README.md
  $<TARGET_FILE_DIR:BodyTrackApp>
)
