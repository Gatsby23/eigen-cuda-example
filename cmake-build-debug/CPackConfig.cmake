# This file will be configured to contain variables for CPack. These variables
# should be set in the CMake list file of the project before CPack module is
# included. The list of available CPACK_xxx variables and their associated
# documentation may be obtained using
#  cpack --help-variable-list
#
# Some variables are common to all generators (e.g. CPACK_PACKAGE_NAME)
# and some are specific to a generator
# (e.g. CPACK_NSIS_EXTRA_INSTALL_COMMANDS). The generator specific variables
# usually begin with CPACK_<GENNAME>_xxxx.


set(CPACK_BUILD_SOURCE_DIRS "/home/robotics/Lab_Study/CUDA_Study/eigen-cuda-example;/home/robotics/Lab_Study/CUDA_Study/eigen-cuda-example/cmake-build-debug")
set(CPACK_CMAKE_GENERATOR "Unix Makefiles")
set(CPACK_COMPONENTS_ALL "Unspecified;eigen;spdlog")
set(CPACK_COMPONENT_UNSPECIFIED_HIDDEN "TRUE")
set(CPACK_COMPONENT_UNSPECIFIED_REQUIRED "TRUE")
set(CPACK_DEBIAN_FILE_NAME "spdlog-1.11.0.deb")
set(CPACK_DEBIAN_PACKAGE_DESCRIPTION "Very fast, header-only/compiled, C++ logging library.")
set(CPACK_DEBIAN_PACKAGE_HOMEPAGE "https://github.com/gabime/spdlog")
set(CPACK_DEBIAN_PACKAGE_NAME "spdlog")
set(CPACK_DEBIAN_PACKAGE_SECTION "libs")
set(CPACK_DEFAULT_PACKAGE_DESCRIPTION_FILE "/snap/clion/250/bin/cmake/linux/x64/share/cmake-3.26/Templates/CPack.GenericDescription.txt")
set(CPACK_DEFAULT_PACKAGE_DESCRIPTION_SUMMARY "IPCToolkit built using CMake")
set(CPACK_GENERATOR "TGZ;ZIP")
set(CPACK_INCLUDE_TOPLEVEL_DIRECTORY "0")
set(CPACK_INSTALL_CMAKE_PROJECTS "/home/robotics/Lab_Study/CUDA_Study/eigen-cuda-example/cmake-build-debug;spdlog;ALL;.")
set(CPACK_INSTALL_PREFIX "/usr/local")
set(CPACK_MODULE_PATH "/home/robotics/Lab_Study/CUDA_Study/eigen-cuda-example/cmake/eigen_cuda_example/;/home/robotics/Lab_Study/CUDA_Study/eigen-cuda-example/cmake/recipes/")
set(CPACK_NSIS_DISPLAY_NAME "IPCToolkit 1.11.0")
set(CPACK_NSIS_INSTALLER_ICON_CODE "")
set(CPACK_NSIS_INSTALLER_MUI_ICON_CODE "")
set(CPACK_NSIS_INSTALL_ROOT "$PROGRAMFILES")
set(CPACK_NSIS_PACKAGE_NAME "IPCToolkit 1.11.0")
set(CPACK_NSIS_UNINSTALL_NAME "Uninstall")
set(CPACK_OBJCOPY_EXECUTABLE "/usr/bin/objcopy")
set(CPACK_OBJDUMP_EXECUTABLE "/usr/bin/objdump")
set(CPACK_OUTPUT_CONFIG_FILE "/home/robotics/Lab_Study/CUDA_Study/eigen-cuda-example/cmake-build-debug/CPackConfig.cmake")
set(CPACK_PACKAGE_CONTACT "Gabi Melman <gmelman1@gmail.com>")
set(CPACK_PACKAGE_DEFAULT_LOCATION "/")
set(CPACK_PACKAGE_DESCRIPTION_FILE "/snap/clion/250/bin/cmake/linux/x64/share/cmake-3.26/Templates/CPack.GenericDescription.txt")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Fast C++ logging library")
set(CPACK_PACKAGE_FILE_NAME "IPCToolkit-1.11.0-Linux")
set(CPACK_PACKAGE_INSTALL_DIRECTORY "IPCToolkit 1.11.0")
set(CPACK_PACKAGE_INSTALL_REGISTRY_KEY "IPCToolkit 1.11.0")
set(CPACK_PACKAGE_NAME "IPCToolkit")
set(CPACK_PACKAGE_RELOCATABLE "true")
set(CPACK_PACKAGE_RELOCATABLE "true")
set(CPACK_PACKAGE_VENDOR "Gabi Melman")
set(CPACK_PACKAGE_VERSION "1.11.0")
set(CPACK_PACKAGE_VERSION_MAJOR "1")
set(CPACK_PACKAGE_VERSION_MINOR "11")
set(CPACK_PACKAGE_VERSION_PATCH "0")
set(CPACK_PROJECT_URL "https://github.com/gabime/spdlog")
set(CPACK_READELF_EXECUTABLE "/usr/bin/readelf")
set(CPACK_RESOURCE_FILE_LICENSE "/snap/clion/250/bin/cmake/linux/x64/share/cmake-3.26/Templates/CPack.GenericLicense.txt")
set(CPACK_RESOURCE_FILE_README "/snap/clion/250/bin/cmake/linux/x64/share/cmake-3.26/Templates/CPack.GenericDescription.txt")
set(CPACK_RESOURCE_FILE_WELCOME "/snap/clion/250/bin/cmake/linux/x64/share/cmake-3.26/Templates/CPack.GenericWelcome.txt")
set(CPACK_RPM_FILE_NAME "spdlog-1.11.0.rpm")
set(CPACK_RPM_PACKAGE_DESCRIPTION "Very fast, header-only/compiled, C++ logging library.")
set(CPACK_RPM_PACKAGE_GROUP "Development/Libraries")
set(CPACK_RPM_PACKAGE_LICENSE "MIT")
set(CPACK_RPM_PACKAGE_NAME "spdlog")
set(CPACK_RPM_PACKAGE_URL "https://github.com/gabime/spdlog")
set(CPACK_SET_DESTDIR "OFF")
set(CPACK_SOURCE_GENERATOR "TBZ2;TGZ;TXZ;TZ")
set(CPACK_SOURCE_OUTPUT_CONFIG_FILE "/home/robotics/Lab_Study/CUDA_Study/eigen-cuda-example/cmake-build-debug/CPackSourceConfig.cmake")
set(CPACK_SOURCE_RPM "OFF")
set(CPACK_SOURCE_TBZ2 "ON")
set(CPACK_SOURCE_TGZ "ON")
set(CPACK_SOURCE_TXZ "ON")
set(CPACK_SOURCE_TZ "ON")
set(CPACK_SOURCE_ZIP "OFF")
set(CPACK_SYSTEM_NAME "Linux")
set(CPACK_THREADS "1")
set(CPACK_TOPLEVEL_TAG "Linux")
set(CPACK_WIX_SIZEOF_VOID_P "8")

if(NOT CPACK_PROPERTIES_FILE)
  set(CPACK_PROPERTIES_FILE "/home/robotics/Lab_Study/CUDA_Study/eigen-cuda-example/cmake-build-debug/CPackProperties.cmake")
endif()

if(EXISTS ${CPACK_PROPERTIES_FILE})
  include(${CPACK_PROPERTIES_FILE})
endif()
