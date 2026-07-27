cmake_minimum_required(VERSION 3.8)
project(bsense_realtime_classifier)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -O2)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclcpp REQUIRED)
find_package(unitree_api REQUIRED)

ament_python_install_package(bsense_classifier PACKAGE_DIR src/bsense_classifier)

add_executable(unitree_ros2_bridge src/unitree_ros2_bridge.cpp)
ament_target_dependencies(unitree_ros2_bridge rclcpp unitree_api)

install(TARGETS unitree_ros2_bridge DESTINATION lib/${PROJECT_NAME})
install(PROGRAMS
  scripts/run_bci_unitree.sh
  tools/mock_bci_sender.py
  tools/mock_eeg_lsl.py
  DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY models DESTINATION share/${PROJECT_NAME})
install(FILES README.md pyproject.toml DESTINATION share/${PROJECT_NAME})

ament_package()
