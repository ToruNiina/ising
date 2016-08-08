set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin/)

add_definitions(-Wall -O3 -std=c++11)

add_executable(simu simulator.cpp)


