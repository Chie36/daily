cmake_minimum_required(VERSION 3.10)
project(CGALExample)

# 设置C++11标准
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# 找到CGAL
find_package(CGAL REQUIRED)

# 包含CGAL头文件目录
include_directories(${CGAL_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# 设置编译选项
set(CMAKE_BUILD_TYPE Release)

# 设置源文件
set(SOURCES test.cpp cgal_example.cpp)

# 创建可执行文件
add_executable(cgal_example ${SOURCES})

# 链接CGAL库
target_link_libraries(cgal_example ${CGAL_LIBRARIES})
