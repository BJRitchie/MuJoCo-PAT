# Findmujoco.cmake
#
# Locates a MuJoCo installation from the precompiled binary release
# (https://github.com/google-deepmind/mujoco/releases). That distribution
# ships libmujoco.so + headers only — it does not export a mujocoConfig.cmake
# package file, so find_package(mujoco REQUIRED) in Config mode fails even
# when MUJOCO_DIR/CMAKE_PREFIX_PATH are set correctly (as setup.sh does).
# This module searches for the library/headers directly instead.
#
# Result variables:
#   mujoco_FOUND
#   MUJOCO_INCLUDE_DIR
#   MUJOCO_LIBRARY
#
# Imported target:
#   mujoco::mujoco

find_path(MUJOCO_INCLUDE_DIR
    NAMES mujoco/mujoco.h
    HINTS "$ENV{MUJOCO_DIR}" "${MUJOCO_DIR}"
    PATH_SUFFIXES include
)

find_library(MUJOCO_LIBRARY
    NAMES mujoco
    HINTS "$ENV{MUJOCO_DIR}" "${MUJOCO_DIR}"
    PATH_SUFFIXES lib
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(mujoco
    REQUIRED_VARS MUJOCO_LIBRARY MUJOCO_INCLUDE_DIR
)

if(mujoco_FOUND AND NOT TARGET mujoco::mujoco)
    add_library(mujoco::mujoco SHARED IMPORTED)
    set_target_properties(mujoco::mujoco PROPERTIES
        IMPORTED_LOCATION "${MUJOCO_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${MUJOCO_INCLUDE_DIR}"
    )
endif()

mark_as_advanced(MUJOCO_INCLUDE_DIR MUJOCO_LIBRARY)
