// Copyright (c) 2022 Samsung Research America, @artofnothingness Alexey Budyakov
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <benchmark/benchmark.h>
#include <string>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <nav_msgs/msg/path.hpp>

#include <nav2_costmap_2d/cost_values.hpp>
#include <nav2_costmap_2d/costmap_2d.hpp>
#include <nav2_costmap_2d/costmap_2d_ros.hpp>
#include <nav2_core/goal_checker.hpp>

#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>

#include "nav2_mppi_controller/motion_models.hpp"
#include "nav2_mppi_controller/controller.hpp"

#include "utils.hpp"

class RosLockGuard
{
public:
  RosLockGuard() {rclcpp::init(0, nullptr);}
  ~RosLockGuard() {rclcpp::shutdown();}
};

RosLockGuard g_rclcpp;

void prepareAndRunBenchmark(
  bool consider_footprint, std::string motion_model,
  std::vector<std::string> critics, benchmark::State & state)
{
  bool visualize = false;

  int batch_size = state.range(0); // Number of steps
  int time_steps = 12;
  unsigned int path_points = 50u;
  int iteration_count = 2;
  double lookahead_distance = 10.0;

  TestCostmapSettings costmap_settings{};
  auto costmap_ros = getDummyCostmapRos(costmap_settings);
  auto costmap = costmap_ros->getCostmap();

  TestPose start_pose = costmap_settings.getCenterPose();
  double path_step = costmap_settings.resolution;

  TestPathSettings path_settings{start_pose, path_points, path_step, path_step};
  TestOptimizerSettings optimizer_settings{batch_size, time_steps, iteration_count,
    lookahead_distance, motion_model, consider_footprint};

  unsigned int offset = 4;
  unsigned int obstacle_size = offset * 2;

  unsigned char obstacle_cost = 250;

  auto [obst_x, obst_y] = costmap_settings.getCenterIJ();

  obst_x = obst_x - offset;
  obst_y = obst_y - offset;
  addObstacle(costmap, {obst_x, obst_y, obstacle_size, obstacle_cost});

  printInfo(optimizer_settings, path_settings, critics);

  rclcpp::NodeOptions options;
  std::vector<rclcpp::Parameter> params;
  setUpControllerParams(visualize, params);
  setUpOptimizerParams(optimizer_settings, critics, params);
  options.parameter_overrides(params);
  auto node = getDummyNode(options);

  auto tf_buffer = std::make_shared<tf2_ros::Buffer>(node->get_clock());
  tf_buffer->setUsingDedicatedThread(true);  // One-thread broadcasting-listening model

  auto broadcaster =
    std::make_shared<tf2_ros::TransformBroadcaster>(node);
  auto tf_listener = std::make_shared<tf2_ros::TransformListener>(*tf_buffer);

  auto map_odom_broadcaster = std::async(
    std::launch::async, sendTf, "map", "odom", broadcaster, node,
    20);

  auto odom_base_link_broadcaster = std::async(
    std::launch::async, sendTf, "odom", "base_link", broadcaster, node,
    20);

  auto controller = getDummyController(node, tf_buffer, costmap_ros);

  // evalControl args
  auto pose = getDummyPointStamped(node, start_pose);
  auto velocity = getDummyTwist();
  auto path = getIncrementalDummyPath(node, path_settings);

  controller->setPlan(path);

  nav2_core::GoalChecker * dummy_goal_checker{nullptr};

  for (auto _ : state) {
    controller->computeVelocityCommands(pose, velocity, dummy_goal_checker);
  }
  map_odom_broadcaster.wait();
  odom_base_link_broadcaster.wait();
}


static void BM_DiffDrive(benchmark::State & state)
{
  bool consider_footprint = true;
  std::string motion_model = "DiffDrive";
  std::vector<std::string> critics = {{"GoalCritic"}, {"GoalAngleCritic"}, {"ObstaclesCritic"},
    {"PathAngleCritic"}, {"PathFollowCritic"}, {"PreferForwardCritic"}};

  prepareAndRunBenchmark(consider_footprint, motion_model, critics, state);
}

BENCHMARK(BM_DiffDrive)->Unit(benchmark::kMillisecond);


BENCHMARK_MAIN();
