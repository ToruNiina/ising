project(Minising)
cmake_minimum_required(VERSION 2.8)

find_package(OpenMP)
if(OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
else()
    message(WARNING "OpenMP not found!")
endif()

# set(CMAKE_CXX_COMPILER clang++-3.6)
add_subdirectory(src)
