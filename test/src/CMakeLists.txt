PROJECT(Joint-VO-SF)

CMAKE_MINIMUM_REQUIRED(VERSION 2.4)
if(COMMAND cmake_policy)
      cmake_policy(SET CMP0003 NEW)  # Required by CMake 2.7+
endif(COMMAND cmake_policy)

# custom cmake modules
LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

#Find dependencies (Eigen is included in MRPT)
FIND_PACKAGE(MRPT REQUIRED base gui opengl obs)
FIND_PACKAGE(OpenCV 3 REQUIRED)
FIND_PACKAGE(TBB REQUIRED)
FIND_PACKAGE(OpenNI2 REQUIRED)

INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/camera)
INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/datasets)
INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/visualization)
INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/segmentation)
INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/Kmeans)
INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/general)
INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR})
INCLUDE_DIRECTORIES(${OpenNI2_INCLUDE_DIRS})
INCLUDE_DIRECTORIES(${TBB_INCLUDE_DIRS})

ADD_LIBRARY(vo_sf_lib
	joint_vo_sf.h
	structs_parallelization.h
        Kmeans/kmeans.cpp
        Kmeans/kmeans.h
        visualization/visualization.cpp
        visualization/visualization.h
        solver.cpp
        segmentation/segmentation_background.cpp
        segmentation/segmentation_background.h
        camera/camera.cpp
        camera/camera.h
        general/general.cpp
        general/general.h
        datasets/datasets.cpp
        datasets/datasets.h
	normal_equation.cpp
        opencv_ext.cpp)
	
TARGET_LINK_LIBRARIES(vo_sf_lib
	${MRPT_LIBS}
	${OpenNI2_LIBRARY}
	${OpenCV_LIBS}
    ${TBB_LIBRARIES})	
			
#To test it with the TUM dataset		
ADD_EXECUTABLE(VO-SF-Datasets 	main_vo_sf_datasets.cpp)
TARGET_LINK_LIBRARIES(VO-SF-Datasets 	vo_sf_lib)


#ADD_EXECUTABLE(VO-SF-Camera 	main_vo_sf_camera.cpp)
#TARGET_LINK_LIBRARIES(VO-SF-Camera 	vo_sf_lib)
# Set optimized building:
IF(CMAKE_COMPILER_IS_GNUCXX)
       SET(CMAKE_BUILD_TYPE "Debug") #I'm not sure if this does anything
     #    SET(CMAKE_BUILD_TYPE "Release") #I'm not sure if this does anything
        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -mtune=native -mavx")
ENDIF(CMAKE_COMPILER_IS_GNUCXX)

