cmake_minimum_required(VERSION 3.10)
project(LAB4OpenCV LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED COMPONENTS core highgui imgproc imgcodecs)

set(SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/source")
set(INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")

function(add_task target_name source_file)
    add_executable(${target_name} "${SOURCE_DIR}/${source_file}" ${ARGN})
    target_include_directories(${target_name} PRIVATE "${INCLUDE_DIR}" ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(${target_name} PRIVATE ${OpenCV_LIBS})
    set_target_properties(${target_name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
endfunction()
add_task(main main.cpp)
