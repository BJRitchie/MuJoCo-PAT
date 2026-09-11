# Findacados.cmake
#
# Locates an acados install built from source (see docker/Dockerfile). acados
# does not ship a reliable acadosConfig.cmake, so this searches the tree
# directly, keyed off the ACADOS_SOURCE_DIR env var (acados' own convention)
# or -DACADOS_DIR=<prefix>.
#
# Result:  acados_FOUND
# Imported target:  acados::acados   (acados + hpipm + blasfeo, with headers)

set(_acados_root "${ACADOS_DIR}")
if(NOT _acados_root)
    set(_acados_root "$ENV{ACADOS_SOURCE_DIR}")
endif()

find_path(ACADOS_INCLUDE_DIR
    NAMES acados_c/ocp_qp_interface.h
    HINTS "${_acados_root}/include"
)
find_library(ACADOS_CORE_LIB    NAMES acados   HINTS "${_acados_root}/lib")
find_library(ACADOS_HPIPM_LIB   NAMES hpipm    HINTS "${_acados_root}/lib")
find_library(ACADOS_BLASFEO_LIB NAMES blasfeo  HINTS "${_acados_root}/lib")

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(acados
    REQUIRED_VARS ACADOS_INCLUDE_DIR ACADOS_CORE_LIB ACADOS_HPIPM_LIB ACADOS_BLASFEO_LIB
    REASON_FAILURE_MESSAGE "set ACADOS_SOURCE_DIR, or build in the container (see docker/)"
)

if(acados_FOUND AND NOT TARGET acados::acados)
    add_library(acados::acados INTERFACE IMPORTED)
    set_target_properties(acados::acados PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES
            "${ACADOS_INCLUDE_DIR};${ACADOS_INCLUDE_DIR}/acados;${ACADOS_INCLUDE_DIR}/blasfeo/include;${ACADOS_INCLUDE_DIR}/hpipm/include"
        # acados core first, then hpipm/blasfeo it depends on.
        INTERFACE_LINK_LIBRARIES
            "${ACADOS_CORE_LIB};${ACADOS_HPIPM_LIB};${ACADOS_BLASFEO_LIB};m"
    )
endif()

mark_as_advanced(ACADOS_INCLUDE_DIR ACADOS_CORE_LIB ACADOS_HPIPM_LIB ACADOS_BLASFEO_LIB)
