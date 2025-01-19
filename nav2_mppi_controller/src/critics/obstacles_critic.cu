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

#include <cmath>
#include "nav2_mppi_controller/critics/obstacles_critic.hpp"

namespace mppi::critics
{

void ObstaclesCritic::initialize()
{
  auto getParam = parameters_handler_->getParamGetter(name_);
  getParam(consider_footprint_, "consider_footprint", false);
  getParam(power_, "cost_power", 1);
  getParam(repulsion_weight_, "repulsion_weight", 1.5);
  getParam(critical_weight_, "critical_weight", 20.0);
  getParam(collision_cost_, "collision_cost", 10000.0);
  getParam(collision_margin_distance_, "collision_margin_distance", 0.10);
  getParam(near_goal_distance_, "near_goal_distance", 0.5);

  collision_checker_.setCostmap(costmap_);
  possibly_inscribed_cost_ = findCircumscribedCost(costmap_ros_);

  if (possibly_inscribed_cost_ < 1.0f) {
    RCLCPP_ERROR(
      logger_,
      "Inflation layer either not found or inflation is not set sufficiently for "
      "optimized non-circular collision checking capabilities. It is HIGHLY recommended to set"
      " the inflation radius to be at MINIMUM half of the robot's largest cross-section. See "
      "github.com/ros-planning/navigation2/tree/main/nav2_smac_planner#potential-fields"
      " for full instructions. This will substantially impact run-time performance.");
  }

  RCLCPP_INFO(
    logger_,
    "ObstaclesCritic instantiated with %d power and %f / %f weights. "
    "Critic will collision check based on %s cost.",
    power_, critical_weight_, repulsion_weight_, consider_footprint_ ?
    "footprint" : "circular");
}

__device__ float findCircumscribedCostDevice(
    float circum_radius,
    float resolution,
    int layer_count,
    float* inflation_layer_costs,
    int* inflation_layer_types) {
  
    static float last_circum_radius = -1.0f;
    static float last_cost = -1.0f;

    if (circum_radius == last_circum_radius) {
        return last_cost; // Early return if footprint size is unchanged
    }

    float result = -1.0f;
    bool inflation_layer_found = false;

    for (int i = 0; i < layer_count; ++i) {
        if (inflation_layer_types[i] != 1) { // Assuming 1 is the inflation layer type
            continue; // Skip non-inflation layers
        }

        inflation_layer_found = true;
        result = inflation_layer_costs[i] * (circum_radius / resolution); // Simplified cost calculation
        break; // Exit the loop after finding the inflation layer
    }

    if (!inflation_layer_found) {
        return -1.0f; // Indicate that no cost could be computed
    }

    last_circum_radius = circum_radius;
    last_cost = result;

    return last_cost;
}

__device__ float distanceToObstacleDevice(float cost, float scale_factor, float min_radius) {
    return (scale_factor * min_radius - log(cost) + log(253.0f)) / scale_factor;
}

__device__ bool inCollisionDevice(float cost, bool consider_footprint) {
    const unsigned char LETHAL_OBSTACLE = 100; // Example value
    const unsigned char INSCRIBED_INFLATED_OBSTACLE = 50; // Example value
    const unsigned char NO_INFORMATION = 255; // Example value

    switch (static_cast<unsigned char>(cost)) {
        case LETHAL_OBSTACLE:
            return true;
        case INSCRIBED_INFLATED_OBSTACLE:
            return !consider_footprint;
        case NO_INFORMATION:
            return false;
    }
    return false;
}

__device__ float costAtPoseDevice(float x, float y, float theta, float resolution, float* point_costs, float* footprint_costs) {
    unsigned int x_i = static_cast<unsigned int>(x / resolution);
    unsigned int y_i = static_cast<unsigned int>(y / resolution);
    
    // Placeholder for collision checking (use point costs for now)
    float cost = point_costs[y_i * 100 + x_i]; // Replace with actual point cost retrieval

    if (footprint_costs != nullptr) {
        float footprint_cost = footprint_costs[y_i * 100 + x_i]; // Replace with actual footprint cost retrieval
        return fminf(cost, footprint_cost);
    }

    return cost;
}

__global__ void scoreKernel(
    float* trajectory_x,
    float* trajectory_y,
    float* trajectory_theta,
    int traj_len,
    float resolution,
    int layer_count,
    float* inflation_layer_costs,
    int* inflation_layer_types,
    float collision_margin_distance,
    float* point_costs,
    float* footprint_costs,
    float* costs) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= traj_len) return;

    float traj_cost = 0.0f;
    float x = trajectory_x[idx];
    float y = trajectory_y[idx];
    float theta = trajectory_theta[idx];

    // Calculate cost at pose
    float cost = costAtPoseDevice(x, y, theta, resolution, point_costs, footprint_costs);

    if (cost < 1.0f) {
        costs[idx] = 0.0f; // In free space
        return;
    }

    // Check for collision
    bool collision = inCollisionDevice(cost, true); // Assume considering footprint

    if (collision) {
        costs[idx] = 10000.0f; // Collision cost
    } else {
        // Calculate distance to obstacle
        float dist_to_obj = distanceToObstacleDevice(cost, 10.0f, 0.5f); // Example scale and min_radius
        if (dist_to_obj < collision_margin_distance) {
            traj_cost += (collision_margin_distance - dist_to_obj);
        }
        costs[idx] = traj_cost; // Store the cost
    }
}

void launchScoreKernel(float* h_trajectory_x, float* h_trajectory_y, float* h_trajectory_theta,
                       int traj_len, float resolution, int layer_count, float* h_inflation_layer_costs,
                       int* h_inflation_layer_types, float collision_margin_distance,
                       float* h_point_costs, float* h_footprint_costs, float* h_costs) {
    // Device pointers
    float *d_trajectory_x, *d_trajectory_y, *d_trajectory_theta;
    float *d_inflation_layer_costs, *d_point_costs, *d_footprint_costs, *d_costs;
    int *d_inflation_layer_types;

    // Allocate device memory
    cudaMalloc(&d_trajectory_x, traj_len * sizeof(float));
    cudaMalloc(&d_trajectory_y, traj_len * sizeof(float));
    cudaMalloc(&d_trajectory_theta, traj_len * sizeof(float));
    cudaMalloc(&d_inflation_layer_costs, layer_count * sizeof(float));
    cudaMalloc(&d_inflation_layer_types, layer_count * sizeof(int));
    cudaMalloc(&d_point_costs, 100 * 100 * sizeof(float));  // Example size for point costs
    cudaMalloc(&d_footprint_costs, 100 * 100 * sizeof(float)); // Example size for footprint costs
    cudaMalloc(&d_costs, traj_len * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_trajectory_x, h_trajectory_x, traj_len * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_trajectory_y, h_trajectory_y, traj_len * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_trajectory_theta, h_trajectory_theta, traj_len * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_inflation_layer_costs, h_inflation_layer_costs, layer_count * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_inflation_layer_types, h_inflation_layer_types, layer_count * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_point_costs, h_point_costs, 100 * 100 * sizeof(float), cudaMemcpyHostToDevice); // Example size
    cudaMemcpy(d_footprint_costs, h_footprint_costs, 100 * 100 * sizeof(float), cudaMemcpyHostToDevice); // Example size

    // Define grid and block size
    int blockSize = 256; // Define a suitable block size
    int numBlocks = (traj_len + blockSize - 1) / blockSize; // Calculate number of blocks

    // Launch the kernel
    scoreKernel<<<numBlocks, blockSize>>>(d_trajectory_x, d_trajectory_y, d_trajectory_theta,
                                           traj_len, resolution, layer_count,
                                           d_inflation_layer_costs, d_inflation_layer_types,
                                           collision_margin_distance, d_point_costs,
                                           d_footprint_costs, d_costs);

    // Check for any errors during kernel launch
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "Kernel launch failed: " << cudaGetErrorString(error) << std::endl;
    }

    // Copy the results back to host
    cudaMemcpy(h_costs, d_costs, traj_len * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_trajectory_x);
    cudaFree(d_trajectory_y);
    cudaFree(d_trajectory_theta);
    cudaFree(d_inflation_layer_costs);
    cudaFree(d_inflation_layer_types);
    cudaFree(d_point_costs);
    cudaFree(d_footprint_costs);
    cudaFree(d_costs);
}

void ObstaclesCritic::score(CriticData & data)
{
    if (!enabled_) {
        return;
    }

    bool near_goal = false;
    if (utils::withinPositionGoalTolerance(near_goal_distance_, data.state.pose.pose, data.path)) {
        near_goal = true;
    }

    // Launch CUDA kernel
    launchScoreKernel(data, collision_cost_, collision_margin_distance_, inflation_radius_,
                      inflation_scale_factor_, critical_weight_, repulsion_weight_, power_);
}

}  // namespace mppi::critics

#include <pluginlib/class_list_macros.hpp>

PLUGINLIB_EXPORT_CLASS(
  mppi::critics::ObstaclesCritic,
  mppi::critics::CriticFunction)
