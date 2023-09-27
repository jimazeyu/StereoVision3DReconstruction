cmake_minimum_required(VERSION 3.0)
project(STEREO_CAMERA)

find_package(Pangolin REQUIRED)
find_package(OpenCV REQUIRED)


include_directories(${Opencv_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})

# 设置可执行文件的所在目录
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/bin)

# 双目相机标定

# 稠密建图
# add_executable(stereoDense stereoDense.cpp include/utils.hpp)
# target_link_libraries(stereoDense ${OpenCV_LIBS} ${Pangolin_LIBRARIES})

# 稀疏建图
add_executable(myStereoSparse stereoSparse.cpp include/utils.hpp include/featureDescriptor.hpp include/featureDetector.hpp include/featureMatch.hpp include/FASTCorner.h include/HarrisCorner.h)
target_link_libraries(myStereoSparse ${OpenCV_LIBS} ${Pangolin_LIBRARIES})
