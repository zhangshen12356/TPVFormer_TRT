# Install script for directory: /home/zs/Code/TPVFormer_TRT/mmdeploy/csrc/mmdeploy/backend_ops/tensorrt

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/home/zs/Code/TPVFormer_TRT/mmdeploy/build/install")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}/home/zs/Code/TPVFormer_TRT/mmdeploy/mmdeploy/lib/libmmdeploy_tensorrt_ops.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/home/zs/Code/TPVFormer_TRT/mmdeploy/mmdeploy/lib/libmmdeploy_tensorrt_ops.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}/home/zs/Code/TPVFormer_TRT/mmdeploy/mmdeploy/lib/libmmdeploy_tensorrt_ops.so"
         RPATH "")
  endif()
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/home/zs/Code/TPVFormer_TRT/mmdeploy/mmdeploy/lib/libmmdeploy_tensorrt_ops.so")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
file(INSTALL DESTINATION "/home/zs/Code/TPVFormer_TRT/mmdeploy/mmdeploy/lib" TYPE MODULE FILES "/home/zs/Code/TPVFormer_TRT/mmdeploy/build/lib/libmmdeploy_tensorrt_ops.so")
  if(EXISTS "$ENV{DESTDIR}/home/zs/Code/TPVFormer_TRT/mmdeploy/mmdeploy/lib/libmmdeploy_tensorrt_ops.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/home/zs/Code/TPVFormer_TRT/mmdeploy/mmdeploy/lib/libmmdeploy_tensorrt_ops.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}/home/zs/Code/TPVFormer_TRT/mmdeploy/mmdeploy/lib/libmmdeploy_tensorrt_ops.so"
         OLD_RPATH "/media/data_12T/zhangshen/TensorRT-8.6.1.6/lib:"
         NEW_RPATH "")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}/home/zs/Code/TPVFormer_TRT/mmdeploy/mmdeploy/lib/libmmdeploy_tensorrt_ops.so")
    endif()
  endif()
endif()

