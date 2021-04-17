
set(SRC_FILE_EXTENSIONS h hpp hu c cpp cu cc)
set(EXCLUDE_FILE_EXTENSTIONS pb.h pb.cc)
set(LINT_DIRS tools)

cmake_policy(SET CMP0009 NEW)  # suppress cmake warning

# find all files of interest
foreach(ext ${SRC_FILE_EXTENSIONS})
    foreach(dir ${LINT_DIRS})
        file(GLOB_RECURSE FOUND_FILES ${CMAKE_SOURCE_DIR}/${dir}/*.${ext})
        set(LINT_SOURCES ${LINT_SOURCES} ${FOUND_FILES})
    endforeach()
endforeach()

set(TOOL_SOURCES ${LINT_SOURCES} PARENT_SCOPE)

