cmake_minimum_required(VERSION 3.10)
project(PolynomialFitting)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(Eigen3 REQUIRED)
# set(EIGEN3_INCLUDE_DIR "/home/lwzy/neolix/gears/x86_64/include/eigen3.2.10")
# include_directories(${EIGEN3_INCLUDE_DIR})
add_executable(polynomial_fitting_test test.cpp polynomial_fitting.cpp)


