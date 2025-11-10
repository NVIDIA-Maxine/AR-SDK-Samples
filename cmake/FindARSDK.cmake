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

# Finds the nvARPose library and its features
#
# And the following targets:
#   nvARPose                - The main ARSDK library
#   NVCVImage               - The NVCVImage library
#   NvCVVolume              - The NvCVVolume library (if available)
#   nvAR<FeatureName>       - For every feature found in the lib directory

if(NOT TARGET nvARPose)
  # Allow the user to specify where to find ARSDK
  set(ARSDK_ROOT "" CACHE PATH "Root directory of AR SDK installation")
  get_filename_component(ARSDK_ROOT "${ARSDK_ROOT}" ABSOLUTE)

  # Find the dynamic library directory
  find_path(nvARPose_LIBRARY_DIR
    NAMES nvARPose.dll libnvARPose.so
    PATHS
    ${ARSDK_ROOT}/bin # If explicitly set
    ${CMAKE_INSTALL_PREFIX}/bin # If installed
    ${CMAKE_CURRENT_SOURCE_DIR}/../bin # If local
    ${ARSDK_ROOT}/lib # Linux, if explicitly set
    ${CMAKE_INSTALL_PREFIX}/lib # Linux, if installed
    ${CMAKE_CURRENT_SOURCE_DIR}/../lib # Linux, if local
    /usr/local/ARSDK/lib/ # Linux, if installed separately
    NO_DEFAULT_PATH
  )

  # Find the model directory
  find_path(nvARPose_MODEL_DIR
    NAMES models
    PATHS
    ${ARSDK_ROOT}/bin # If explicitly set
    ${CMAKE_INSTALL_PREFIX}/bin # If installed
    ${CMAKE_CURRENT_SOURCE_DIR}/../bin # If local
    ${ARSDK_ROOT}/lib # Linux, if explicitly set
    ${CMAKE_INSTALL_PREFIX}/lib # Linux, if installed
    ${CMAKE_CURRENT_SOURCE_DIR}/../lib # Linux, if local
    /usr/local/ARSDK/lib/ # Linux, if installed separately
    NO_DEFAULT_PATH
  )
  set(nvARPose_MODEL_DIR ${nvARPose_MODEL_DIR}/models)

  if(NOT ${ARSDK_ROOT} AND DEFINED nvARPose_LIBRARY_DIR AND EXISTS ${nvARPose_LIBRARY_DIR})
    set(ARSDK_ROOT "${nvARPose_LIBRARY_DIR}/../")
    get_filename_component(ARSDK_ROOT "${ARSDK_ROOT}" ABSOLUTE)
    set(ARSDK_ROOT "${ARSDK_ROOT}" CACHE PATH "Root directory of AR SDK installation" FORCE)
  endif()

  message(STATUS "ARSDK_ROOT: ${ARSDK_ROOT}")

  # Find the import library file for nvARPose
  find_library(nvARPose_IMPORT_LIBRARY
    NAMES nvARPose.lib libnvARPose.so
    PATHS
    ${ARSDK_ROOT}/lib
    ${ARSDK_ROOT}/bin
    NO_DEFAULT_PATH
  )

  # Find the import library file for NVCVImage
  find_library(NVCVImage_IMPORT_LIBRARY
    NAMES NVCVImage.lib libNVCVImage.so
    PATHS
    ${ARSDK_ROOT}/lib
    ${ARSDK_ROOT}/bin
    NO_DEFAULT_PATH
  )

  # Find the import library file for NvCVVolume
  find_library(NvCVVolume_IMPORT_LIBRARY
    NAMES NvCVVolume.lib libNvCVVolume.so
    PATHS
    ${ARSDK_ROOT}/lib
    ${ARSDK_ROOT}/bin
    NO_DEFAULT_PATH
  )

  # Find CUDA libraries and include files
  if(NOT TARGET CUDA_LIBS)
    set(CUDA_LIB_NAMES
      cuda
      cudart
      curand
      nppc
      nppif
      nppig
    )

    set(CUDA_LIB_VARIABLES
      CUDA_CUDA_LIBRARY
      CUDA_CUDART_LIBRARY
      CUDA_CURAND_LIBRARY
      CUDA_NPPC_LIBRARY
      CUDA_NPPIF_LIBRARY
      CUDA_NPPIG_LIBRARY
    )

    set(CUDA_LIB_PATHS
      ${ARSDK_ROOT}/external/cuda/lib
      ${ARSDK_ROOT}/lib
    )

    # Find all CUDA libraries using a loop
    list(LENGTH CUDA_LIB_NAMES NUM_LIBS)
    math(EXPR LAST_INDEX "${NUM_LIBS} - 1")
    foreach(INDEX RANGE ${LAST_INDEX})
      list(GET CUDA_LIB_NAMES ${INDEX} LIB_NAME)
      list(GET CUDA_LIB_VARIABLES ${INDEX} VAR_NAME)
      find_library(${VAR_NAME}
        NAMES ${LIB_NAME}.lib lib${LIB_NAME}.so
        PATHS ${CUDA_LIB_PATHS}
        NO_DEFAULT_PATH
      )
    endforeach()

    # Find CUDA include directory
    find_path(CUDA_INCLUDE_DIR
      NAMES cuda.h
      PATHS
      ${ARSDK_ROOT}/external/cuda/include
      NO_DEFAULT_PATH
    )

    # Create CUDA target if libraries are found
    if(CUDA_CUDA_LIBRARY OR CUDA_CUDART_LIBRARY OR CUDA_CURAND_LIBRARY OR CUDA_NPPC_LIBRARY OR CUDA_NPPIF_LIBRARY OR CUDA_NPPIG_LIBRARY)
      if(NOT TARGET CUDA_LIBS)
        add_library(CUDA_LIBS INTERFACE)
      endif()

      # Link against available CUDA libraries
      if(CUDA_CUDA_LIBRARY)
        target_link_libraries(CUDA_LIBS INTERFACE ${CUDA_CUDA_LIBRARY})
      endif()
      if(CUDA_CUDART_LIBRARY)
        target_link_libraries(CUDA_LIBS INTERFACE ${CUDA_CUDART_LIBRARY})
      endif()
      if(CUDA_CURAND_LIBRARY)
        target_link_libraries(CUDA_LIBS INTERFACE ${CUDA_CURAND_LIBRARY})
      endif()
      if(CUDA_NPPC_LIBRARY)
        target_link_libraries(CUDA_LIBS INTERFACE ${CUDA_NPPC_LIBRARY})
      endif()
      if(CUDA_NPPIF_LIBRARY)
        target_link_libraries(CUDA_LIBS INTERFACE ${CUDA_NPPIF_LIBRARY})
      endif()
      if(CUDA_NPPIG_LIBRARY)
        target_link_libraries(CUDA_LIBS INTERFACE ${CUDA_NPPIG_LIBRARY})
      endif()

    endif()
  endif()

  if(NOT nvARPose_IMPORT_LIBRARY)
    add_library(nvARPose STATIC
      ${ARSDK_ROOT}/nvar/src/nvARProxy.cpp
      ${ARSDK_ROOT}/nvar/include/nvAR.h
      ${ARSDK_ROOT}/nvar/include/nvAR_defs.h
      ${ARSDK_ROOT}/nvar/include/nvCVStatus.h)
    target_include_directories(nvARPose PUBLIC ${ARSDK_ROOT}/nvar/include)
    set(nvARPose_IMPORT_LIBRARY nvARPose)
  endif()

  if(NOT NVCVImage_IMPORT_LIBRARY)
    add_library(NVCVImage STATIC
      ${ARSDK_ROOT}/nvar/src/nvCVImageProxy.cpp
      ${ARSDK_ROOT}/nvar/include/nvCVImage.h
      ${ARSDK_ROOT}/nvar/include/nvCVStatus.h)
    target_include_directories(NVCVImage PUBLIC ${ARSDK_ROOT}/nvar/include)
    set(NVCVImage_IMPORT_LIBRARY NVCVImage)
  endif()

  if(NvCVVolume_IMPORT_LIBRARY)
    add_library(NvCVVolume SHARED IMPORTED)
  elseif(EXISTS "${ARSDK_ROOT}/nvar/src/nvCVVolumeProxy.cpp")
    add_library(NvCVVolume STATIC
      ${ARSDK_ROOT}/nvar/src/nvCVVolumeProxy.cpp
      ${ARSDK_ROOT}/nvar/include/nvCVTriplaneVolume.h
      ${ARSDK_ROOT}/nvar/include/nvCVVolumeDefs.h)
    target_include_directories(NvCVVolume PUBLIC ${ARSDK_ROOT}/nvar/include)
    set(NvCVVolume_IMPORT_LIBRARY NvCVVolume)
  endif()

  # Find the include directory
  find_path(nvARPose_INCLUDE_DIR
    NAMES nvAR.h
    PATHS
    ${ARSDK_ROOT}/include
    ${ARSDK_ROOT}/nvar/include
    NO_DEFAULT_PATH
  )

  # Set version from header if available
  if(nvARPose_INCLUDE_DIR)
    # Find all version*.h files
    file(GLOB VERSION_FILES "${ARSDK_ROOT}/arsdk_version.h")

    # On Linux, also check in share directory if not found in root
    if(NOT VERSION_FILES AND UNIX)
      file(GLOB VERSION_FILES "${ARSDK_ROOT}/share/arsdk_version.h")
    endif()

    if(VERSION_FILES)
      list(GET VERSION_FILES 0 VERSION_FILE)
      # Sort to ensure consistent order, and read the first one
      list(SORT VERSION_FILES)
      file(READ "${VERSION_FILE}" VERSION_HEADER)

      string(REGEX MATCH "NVIDIA_AR_SDK_VERSION_MAJOR[ \t]+([0-9]+)" _ "${VERSION_HEADER}")
      set(nvARPose_VERSION_VERSION_MAJOR "${CMAKE_MATCH_1}")
      string(REGEX MATCH "NVIDIA_AR_SDK_VERSION_MINOR[ \t]+([0-9]+)" _ "${VERSION_HEADER}")
      set(nvARPose_VERSION_VERSION_MINOR "${CMAKE_MATCH_1}")
      string(REGEX MATCH "NVIDIA_AR_SDK_VERSION_RELEASE[ \t]+([0-9]+)" _ "${VERSION_HEADER}")
      set(nvARPose_VERSION_VERSION_RELEASE "${CMAKE_MATCH_1}")
      string(REGEX MATCH "NVIDIA_AR_SDK_VERSION_BUILD[ \t]+([0-9]+)" _ "${VERSION_HEADER}")
      set(nvARPose_VERSION_VERSION_BUILD "${CMAKE_MATCH_1}")
      set(nvARPose_VERSION_STRING "${nvARPose_VERSION_VERSION_MAJOR}.${nvARPose_VERSION_VERSION_MINOR}.${nvARPose_VERSION_VERSION_RELEASE}.${nvARPose_VERSION_VERSION_BUILD}")
    endif()
  endif()

  if(NOT nvARPose_VERSION_STRING)
    message(WARNING "Unable to deduce SDK version")
    set(nvARPose_VERSION_STRING "")
  endif()

  # Create targets
  include(FindPackageHandleStandardArgs)

  find_package_handle_standard_args(ARSDK
    FAIL_MESSAGE
    "ARSDK not found. Please set ARSDK_ROOT to the root directory of the AR SDK installation."
    REQUIRED_VARS
    nvARPose_IMPORT_LIBRARY
    NVCVImage_IMPORT_LIBRARY
    nvARPose_INCLUDE_DIR
    VERSION_VAR nvARPose_VERSION_STRING
  )

  if(ARSDK_FOUND)
    # Main library target
    if(NOT TARGET nvARPose)
      add_library(nvARPose SHARED IMPORTED)
    endif()
    set_target_properties(nvARPose PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${nvARPose_INCLUDE_DIR}"
      IMPORTED_IMPLIB "${nvARPose_IMPORT_LIBRARY}"
      IMPORTED_LOCATION "${nvARPose_IMPORT_LIBRARY}"
      DYNAMIC_LIBRARY_DIR "${nvARPose_LIBRARY_DIR}"
      MODEL_DIR "${nvARPose_MODEL_DIR}"
    )

    # NvCVImage library target
    if(NOT TARGET NVCVImage)
      add_library(NVCVImage SHARED IMPORTED)
    endif()
    set_target_properties(NVCVImage PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${nvARPose_INCLUDE_DIR}"
      IMPORTED_IMPLIB "${NVCVImage_IMPORT_LIBRARY}"
      IMPORTED_LOCATION "${NVCVImage_IMPORT_LIBRARY}"
      DYNAMIC_LIBRARY_DIR "${nvARPose_LIBRARY_DIR}"
    )

    # NvCVVolume library target
    if(TARGET NvCVVolume)
      set_target_properties(NvCVVolume PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${nvARPose_INCLUDE_DIR}"
        DYNAMIC_LIBRARY_DIR "${nvARPose_LIBRARY_DIR}"
      )
      if(NvCVVolume_IMPORT_LIBRARY)
        set_target_properties(NvCVVolume PROPERTIES
          IMPORTED_IMPLIB "${NvCVVolume_IMPORT_LIBRARY}"
          IMPORTED_LOCATION "${NvCVVolume_IMPORT_LIBRARY}"
        )
      endif()
    endif()

    # Dynamically discover feature libraries
    file(GLOB DIR_LIST LIST_DIRECTORIES true "${ARSDK_ROOT}/features/*")
    set(FEATURE_FOLDERS "")
    foreach(DIR_OBJ ${DIR_LIST})
      if(IS_DIRECTORY ${DIR_OBJ})
        list(APPEND FEATURE_FOLDERS ${DIR_OBJ})
      endif()
    endforeach()
    foreach(FEATURE_FOLDER ${FEATURE_FOLDERS})
      file(GLOB HEADERS "${FEATURE_FOLDER}/include/*.h")
      foreach(HEADER ${HEADERS})
        get_filename_component(HEADER_NAME ${HEADER} NAME_WE)
        get_filename_component(FEATURE_FOLDER_NAME ${FEATURE_FOLDER} NAME)
        string(TOLOWER "${HEADER_NAME}" HEADER_NAME_LOWER)
        if("${HEADER_NAME_LOWER}" STREQUAL "${FEATURE_FOLDER_NAME}")
          set(FEATURE_NAME ${HEADER_NAME})
        endif()
      endforeach()
      if(NOT FEATURE_NAME)
        message(FATAL_ERROR "Invalid feature. Header not matching feature name: ${FEATURE_FOLDER}")
      endif()
      # Create interface library for the feature
      add_library(${FEATURE_NAME} INTERFACE)

      # Read the version string from the header file and set the version property
      file(READ "${FEATURE_FOLDER}/include/${FEATURE_NAME}.h" FEATURE_HEADER)
      string(TOUPPER "${FEATURE_NAME}" PROJECT_NAME_UPPER)
      string(REGEX MATCH "#define[ \t]+${PROJECT_NAME_UPPER}_VERSION[ \t]+\"([^\"]+)\"" VERSION_MATCH "${FEATURE_HEADER}")
      set(FEATURE_VERSION_STRING "${CMAKE_MATCH_1}")
      set_target_properties(${FEATURE_NAME} PROPERTIES
        INTERFACE_VERSION ${FEATURE_VERSION_STRING}
      )

      target_include_directories(${FEATURE_NAME} INTERFACE ${FEATURE_FOLDER}/include)
      if(UNIX)
        set_target_properties(${FEATURE_NAME} PROPERTIES INTERFACE_DYNAMIC_LIBRARY_DIRECTORY ${FEATURE_FOLDER}/lib)
      else()
        set_target_properties(${FEATURE_NAME} PROPERTIES INTERFACE_DYNAMIC_LIBRARY_DIRECTORY ${FEATURE_FOLDER}/bin)
      endif()
    endforeach()
  endif()
endif()
