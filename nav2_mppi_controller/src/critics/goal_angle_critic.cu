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

#include "nav2_mppi_controller/critics/goal_angle_critic.hpp"
#include <cuda_runtime.h>
#include <memory>
#include <device_launch_parameters.h>
#include <xtensor/xadapt.hpp>

namespace mppi::critics
{

__global__ void computeGoalAngleKernel(
    const float* yaws, float goal_yaw, float weight, int power, int n, float* costs)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float shortest_distance = fmodf(fabsf(yaws[idx] - goal_yaw) + M_PI, 2 * M_PI) - M_PI;
        float cost = powf(fabsf(shortest_distance) * weight, power);
        atomicAdd(costs, cost); // Sumowanie kosztów (atomowe, by uniknąć konfliktów).
    }
}


void GoalAngleCritic::initialize()
{
  auto getParam = parameters_handler_->getParamGetter(name_);

  getParam(power_, "cost_power", 1);
  getParam(weight_, "cost_weight", 3.0);

  getParam(threshold_to_consider_, "threshold_to_consider", 0.5);

  RCLCPP_INFO(
    logger_,
    "GoalAngleCritic instantiated with %d power, %f weight, and %f "
    "angular threshold.",
    power_, weight_, threshold_to_consider_);
}

void GoalAngleCritic::score(CriticData & data)
{
    if (!enabled_ || !utils::withinPositionGoalTolerance(
            threshold_to_consider_, data.state.pose.pose, data.path))
    {
        return;
    }

    const auto goal_idx = data.path.x.shape(0) - 1;
    const float goal_yaw = data.path.yaws(goal_idx);

    // Liczba trajektorii.
    int n = data.trajectories.yaws.size();

    // Alokacja pamięci na GPU.
    float* d_yaws;
    float* d_costs;
    cudaMalloc(&d_yaws, n * sizeof(float));
    cudaMalloc(&d_costs, sizeof(float));
    cudaMemset(d_costs, 0, sizeof(float));

    // Kopiowanie danych na GPU.
    cudaMemcpy(d_yaws, data.trajectories.yaws.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    // Uruchamianie kernela.
    int threads_per_block = 256;
    int num_blocks = (n + threads_per_block - 1) / threads_per_block;
    computeGoalAngleKernel<<<num_blocks, threads_per_block>>>(
        d_yaws, goal_yaw, weight_, power_, n, d_costs);

    // Kopiowanie wyników z GPU na CPU.
    float total_cost;
    cudaMemcpy(&total_cost, d_costs, sizeof(float), cudaMemcpyDeviceToHost);

    // Dodanie kosztu do całkowitych kosztów.
    data.costs += total_cost;

    // Zwolnienie pamięci GPU.
    cudaFree(d_yaws);
    cudaFree(d_costs);
}
}  // namespace mppi::critics

#include <pluginlib/class_list_macros.hpp>

PLUGINLIB_EXPORT_CLASS(
  mppi::critics::GoalAngleCritic,
  mppi::critics::CriticFunction)
