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

// CUDA includes
#include <cuda_runtime.h>
#include <cmath>

class RosLockGuard
{
public:
  RosLockGuard() {rclcpp::init(0, nullptr);}
  ~RosLockGuard() {rclcpp::shutdown();}
};

RosLockGuard g_rclcpp;

// CUDA kernel to compute goal_critic in parallel
__global__ void goal_critic_kernel(const float* traj_x, const float* traj_y, float goal_x, float goal_y, float* dists, int num_trajectories)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_trajectories) {
        float dx = traj_x[idx] - goal_x;
        float dy = traj_y[idx] - goal_y;
        dists[idx] = sqrt(dx * dx + dy * dy);
    }
}

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

  // Allocate memory for GPU
  int num_trajectories = batch_size; // Number of trajectories
  float *d_traj_x, *d_traj_y, *d_dists;
  cudaMalloc(&d_traj_x, num_trajectories * sizeof(float));
  cudaMalloc(&d_traj_y, num_trajectories * sizeof(float));
  cudaMalloc(&d_dists, num_trajectories * sizeof(float));

  // Initialize dummy trajectory data (example, you need to fill this with actual trajectory data)
  float* h_traj_x = new float[num_trajectories];
  float* h_traj_y = new float[num_trajectories];
  for (int i = 0; i < num_trajectories; ++i) {
    h_traj_x[i] = i * 0.1f;  // Example data
    h_traj_y[i] = i * 0.2f;  // Example data
  }

  // Copy trajectory data to GPU
  cudaMemcpy(d_traj_x, h_traj_x, num_trajectories * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_traj_y, h_traj_y, num_trajectories * sizeof(float), cudaMemcpyHostToDevice);

  float goal_x = 10.0f;  // Example goal x
  float goal_y = 10.0f;  // Example goal y

  std::vector<float> dists(num_trajectories);

  for (auto _ : state) {
    int blockSize = 256;
    int numBlocks = (num_trajectories + blockSize - 1) / blockSize;

    std::cout << "Launching goal_critic_kernel with " << numBlocks << " blocks and " << blockSize << " threads per block" << std::endl;

    // run kernel
    goal_critic_kernel<<<numBlocks, blockSize>>>(d_traj_x, d_traj_y, goal_x, goal_y, d_dists, num_trajectories);

    // wait for it to finish
    cudaDeviceSynchronize();

    // copy data to host
    cudaMemcpy(dists.data(), d_dists, num_trajectories * sizeof(float), cudaMemcpyDeviceToHost);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    }
  }

  cudaFree(d_traj_x);
  cudaFree(d_traj_y);
  cudaFree(d_dists);

  map_odom_broadcaster.wait();
  odom_base_link_broadcaster.wait();
}

static void BM_DiffDrive(benchmark::State & state)
{
  bool consider_footprint = true;
  std::string motion_model = "DiffDrive";
  std::vector<std::string> critics = {{"GoalCritic"}};
  prepareAndRunBenchmark(consider_footprint, motion_model, critics, state);
}

BENCHMARK(BM_DiffDrive)->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
