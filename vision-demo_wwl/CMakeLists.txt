cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project( vision-demo )

set( INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include )
set( SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src )
set( APP_DIR ${CMAKE_CURRENT_SOURCE_DIR}/app )
set( THIRD_PARTY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/KCFcpp/src )

find_package( OpenCV REQUIRED )
find_package( Caffe )

add_definitions("-std=c++0x -O3")

include_directories( ${INCLUDE_DIR} ${Caffe_INCLUDE_DIRS} ${THIRD_PARTY_DIR} )
aux_source_directory( ${SRC_DIR} SRC_LIST )
add_definitions( ${Caffe_DEFINITIONS} )
#add_definitions( -DCPU_ONLY )

aux_source_directory( ${APP_DIR} APP_LIST )
aux_source_directory( ${THIRD_PARTY_DIR} THIRD_PARTY_LIST )
add_library( vision-demo SHARED ${SRC_LIST} ${THIRD_PARTY_LIST} ) 

add_executable( video_demo ${APP_DIR}/video_demo.cpp ${INCLUDE_DIR} ${SRC_DIR} )
add_executable( test_segmentation ${APP_DIR}/test_segmentation.cpp ${INCLUDE_DIR} ${SRC_DIR} )
add_executable( test_detection ${APP_DIR}/test_detection.cpp ${INCLUDE_DIR} ${SRC_DIR} )
add_executable( test_tracker ${APP_DIR}/test_tracker.cpp ${INCLUDE_DIR} ${SRC_DIR} )

target_link_libraries( video_demo vision-demo ${OpenCV_LIBS} ${Caffe_LIBRARIES} )
target_link_libraries( test_segmentation vision-demo ${OpenCV_LIBS} ${Caffe_LIBRARIES} )
target_link_libraries( test_detection vision-demo ${OpenCV_LIBS} ${Caffe_LIBRARIES} )
target_link_libraries( test_tracker vision-demo ${OpenCV_LIBS} ${Caffe_LIBRARIES} )
