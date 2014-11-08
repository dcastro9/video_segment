find_package(Boost COMPONENTS system thread REQUIRED)
find_package(FFMPEG REQUIRED)
find_package(GLog REQUIRED)
find_package(OpenCV2 REQUIRED)
find_package(ZLIB REQUIRED)

set(DEPENDENT_PACKAGES base imagefilter)

set(DEPENDENT_INCLUDES ${OpenCV2_INCLUDE_DIRS}
                       ${Boost_INCLUDE_DIR}
                       ${FFMPEG_INCLUDE_DIR}
                       ${GLOG_INCLUDE_DIR}
                       )

set(DEPENDENT_LIBRARIES ${OpenCV2_LIBRARIES}
                        ${Boost_LIBRARIES}
                        ${FFMPEG_LIBRARIES}
                        ${GLOG_LIBRARIES}
                        ${ZLIB_LIBRARIES})

set(CREATED_PACKAGES optical_flow)
