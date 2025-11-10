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

#########################
# SpeechLivePortraitApp #
#########################

set(REQUIRED_FEATURES nvARFaceExpressions:1.0.0.0 nvARLandmarkDetection:1.0.0.0 nvARFaceBoxDetection:1.0.0.0 nvARFace3DReconstruction:1.0.0.0)
foreach(REQUIRED_FEATURE_AND_VERSION IN LISTS REQUIRED_FEATURES)
  verify_app_feature_dependency(${REQUIRED_FEATURE_AND_VERSION} RESULT)
  if(NOT RESULT)
    message(${SAMPLES_DEPS_ERROR_LEVEL} "Skipping SpeechLivePortraitApp.")
    return()
  endif()
endforeach()

if(WIN32)
  set(RUN_FILES
    run_speechliveportrait.bat
  )
else()
  set(RUN_FILES
    run_speechliveportrait.sh
    setup_env.sh)
endif()
set(APP_SRCS
  SpeechLivePortraitApp.cpp
  ${ARSDKSampleApps_utils_DIR}/waveReadWrite.cpp
)
add_executable(SpeechLivePortraitApp
  README.md
  ${APP_SRCS}
)
if(TARGET OpenCV)
  set(OPENCV OpenCV)
else()
  set(OPENCV opencv346)
endif()
#ifdef ENABLE_INTERNAL_API  // We use unifdef on public CMakeLists.txt
if(TARGET nvARPoseIncludes)
  target_link_libraries(SpeechLivePortraitApp PRIVATE nvARPoseIncludes)
endif()
#endif  // ENABLE_INTERNAL_API

target_link_libraries(SpeechLivePortraitApp PRIVATE
  GLM
  ${OPENCV}
  nvARPose
  NVCVImage
)
if(UNIX)
  target_link_libraries(SpeechLivePortraitApp PRIVATE dl)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fpermissive")
endif()
target_include_directories(SpeechLivePortraitApp PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${ARSDKSampleApps_utils_DIR}
  ${OpenCV_INCLUDE_DIRS}
)

if(TARGET CUDA_LIBS)
  target_link_libraries(SpeechLivePortraitApp PRIVATE ${CUDA_CUDART_LIB} ${CUDA_CUDA_LIB} ${CUDA_NPPC_LIB} ${CUDA_NPPIG_LIB})
  if(CUDA_INCLUDE_DIR)
    target_include_directories(SpeechLivePortraitApp PRIVATE ${CUDA_INCLUDE_DIR})
  endif()
elseif(CUDA_LIBS)
  # if CUDA_LIBS is a variable list, not a target
  target_link_libraries(SpeechLivePortraitApp PRIVATE ${CUDA_LIBS})
  if(CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES)
    target_include_directories(SpeechLivePortraitApp PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
  endif()
else()
  message(FATAL_ERROR "CUDA_LIBS not found (neither as target nor as variable)")
endif()

get_target_property(NVAR_MODEL_DIR nvARPose MODEL_DIR)
get_target_property(NVAR_DYNAMIC_LIBRARY_DIR nvARPose DYNAMIC_LIBRARY_DIR)
set(NVAR_DYNAMIC_LIBRARY_DIRS ${NVAR_DYNAMIC_LIBRARY_DIR})

set(REQUIRED_FEATURES nvARSpeechLivePortrait nvARFace3DReconstruction nvARLandmarkDetection nvARFaceBoxDetection)
foreach(REQUIRED_FEATURE IN LISTS REQUIRED_FEATURES)
  if(NOT TARGET ${REQUIRED_FEATURE})
    message(WARNING "Required feature ${REQUIRED_FEATURE} is not available. Skipping SpeechLivePortraitApp.")
    set_target_properties(SpeechLivePortraitApp PROPERTIES EXCLUDE_FROM_ALL 1 EXCLUDE_FROM_DEFAULT_BUILD 1)
    return()
  endif()
  target_link_libraries(SpeechLivePortraitApp PRIVATE ${REQUIRED_FEATURE})
  get_target_property(FEATURE_DYNAMIC_LIBRARY_DIR ${REQUIRED_FEATURE} INTERFACE_DYNAMIC_LIBRARY_DIRECTORY)
  if(FEATURE_DYNAMIC_LIBRARY_DIR)
    list(APPEND NVAR_DYNAMIC_LIBRARY_DIRS ${FEATURE_DYNAMIC_LIBRARY_DIR})
  endif()
endforeach()

if(WIN32)
  get_target_property(OPENCV_PATH_STR ${OPENCV} DYNAMIC_LIBRARY_DIR)
  #ifdef ENABLE_INTERNAL_API  // We use unifdef on public CMakeLists.txt
  if(NOT OPENCV_PATH_STR)
    set(OPENCV_PATH_STR ${CVVR_EXT_DIR}/opencv/build/x64/vc15/bin)
  endif()
  #endif  // ENABLE_INTERNAL_API

  set(PATH_STR "PATH=%PATH%" ${OPENCV_PATH_STR};${NVAR_DYNAMIC_LIBRARY_DIRS})
  set(CMD_ARG_STR "--model_path=\"${NVAR_MODEL_DIR}\" --in_src=speech_live_portrait_sample_portrait.png --in_drv=speech_live_portrait_sample_audio.wav --out=speech_live_portrait_output.mp4 --capture_outputs=true")
  set_target_properties(SpeechLivePortraitApp PROPERTIES
    FOLDER SampleApps
    VS_DEBUGGER_ENVIRONMENT "${PATH_STR}"
    VS_DEBUGGER_COMMAND_ARGUMENTS "${CMD_ARG_STR}"
    VS_DEBUGGER_WORKING_DIRECTORY "$<TARGET_FILE_DIR:SpeechLivePortraitApp>"
  )
  # https://stackoverflow.com/questions/28178978/how-to-generate-pdb-files-for-release-build-with-cmake-flags
  target_compile_options(SpeechLivePortraitApp PRIVATE $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CXX>>:/Zi>) # Keep flag specific to C++ to avoid Cuda build errors
  target_link_options(SpeechLivePortraitApp PRIVATE $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CXX>>:/INCREMENTAL:NO;/DEBUG;/OPT:REF;/OPT:ICF>)
endif()
# Configure run-files
foreach(RUN_FILE IN LISTS RUN_FILES)
  # Note: Separate configure_file from file copy for older CMake version to support file permissions
  configure_file(${RUN_FILE}.in ${CMAKE_CURRENT_BINARY_DIR}/tmp/${RUN_FILE} @ONLY)
  file(COPY ${CMAKE_CURRENT_BINARY_DIR}/tmp/${RUN_FILE} DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/run_files FILE_PERMISSIONS OWNER_EXECUTE OWNER_READ OWNER_WRITE)
  file(REMOVE_RECURSE ${CMAKE_CURRENT_BINARY_DIR}/tmp/)
  add_custom_command(TARGET SpeechLivePortraitApp POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_CURRENT_BINARY_DIR}/run_files/${RUN_FILE}
    $<TARGET_FILE_DIR:SpeechLivePortraitApp>)
endforeach()
# Add resources and README.md file
add_custom_command(TARGET SpeechLivePortraitApp POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
  ${ARSDKSampleApps_RESOURCES_DIR}/speech_live_portrait_sample_portrait.png
  ${ARSDKSampleApps_RESOURCES_DIR}/speech_live_portrait_sample_audio.wav
  ${CMAKE_CURRENT_SOURCE_DIR}/README.md
  $<TARGET_FILE_DIR:SpeechLivePortraitApp>
)
