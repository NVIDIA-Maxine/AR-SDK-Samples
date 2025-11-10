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

############################
# FaceTrackTritonClientApp #
############################

set(RUN_FILES
  run_facetracktritonclientapp_offline.sh
  setup_env.sh)

set(FACETRACKTRITONCLIENT_SRCS
  FaceTrackTritonClientApp.cpp
  ${ARSDKSampleApps_utils_DIR}/batchUtilities.cpp
  ${ARSDKSampleApps_utils_DIR}/batchUtilities.h
)
add_executable(FaceTrackTritonClientApp README.md ${FACETRACKTRITONCLIENT_SRCS})

if(TARGET OpenCV)
  set(OPENCV OpenCV)
else()
  set(OPENCV opencv346)
endif()

target_link_libraries(FaceTrackTritonClientApp PRIVATE
  ${OPENCV}
  ${CMAKE_DL_LIBS}
  nvARPose
  NVCVImage
)

target_include_directories(FaceTrackTritonClientApp PRIVATE
  ${ARSDKSampleApps_utils_DIR}
  ${OpenCV_INCLUDE_DIRS}
)

# Link to feature interface targets if available (from FindARSDK.cmake)
if(TARGET nvARFaceBoxDetection)
  target_link_libraries(FaceTrackTritonClientApp PRIVATE nvARFaceBoxDetection)
endif()
if(TARGET nvARLandmarkDetection)
  target_link_libraries(FaceTrackTritonClientApp PRIVATE nvARLandmarkDetection)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fpermissive")

get_target_property(NVAR_DYNAMIC_LIBRARY_DIR nvARPose DYNAMIC_LIBRARY_DIR)
get_target_property(NVAR_MODEL_DIR nvARPose MODEL_DIR)

# Configure run-files
foreach(RUN_FILE IN LISTS RUN_FILES)
  # Note: Separate configure_file from file copy for older CMake version to support file permissions
  configure_file(${RUN_FILE}.in ${CMAKE_CURRENT_BINARY_DIR}/tmp/${RUN_FILE} @ONLY)
  file(COPY ${CMAKE_CURRENT_BINARY_DIR}/tmp/${RUN_FILE} DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/run_files FILE_PERMISSIONS OWNER_EXECUTE OWNER_READ OWNER_WRITE)
  file(REMOVE_RECURSE ${CMAKE_CURRENT_BINARY_DIR}/tmp/)
  add_custom_command(TARGET FaceTrackTritonClientApp POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_CURRENT_BINARY_DIR}/run_files/${RUN_FILE}
    $<TARGET_FILE_DIR:FaceTrackTritonClientApp>)
endforeach()

# Add resources and README.md file
add_custom_command(TARGET FaceTrackTritonClientApp POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
  ${ARSDKSampleApps_RESOURCES_DIR}/faceTrack.mp4
  ${CMAKE_CURRENT_SOURCE_DIR}/README.md
  $<TARGET_FILE_DIR:FaceTrackTritonClientApp>
)
