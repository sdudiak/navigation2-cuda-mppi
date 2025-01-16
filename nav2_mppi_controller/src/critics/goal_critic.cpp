#include "nav2_mppi_controller/critics/goal_critic.hpp"
#include <cuda_runtime.h>
#include <memory>

namespace mppi::critics
{
using xt::evaluation_strategy::immediate;

// CUDA kernel for distance computation
__global__ void computeDistancesKernel(const float* traj_x, const float* traj_y, float goal_x, float goal_y,
                                       float* dists, size_t num_trajectories, size_t num_points)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_trajectories * num_points)
    {
        int traj_idx = idx / num_points;
        int point_idx = idx % num_points;
        float dx = traj_x[traj_idx * num_points + point_idx] - goal_x;
        float dy = traj_y[traj_idx * num_points + point_idx] - goal_y;
        dists[traj_idx * num_points + point_idx] = sqrtf(dx * dx + dy * dy);
    }
}

void GoalCritic::initialize()
{
    auto getParam = parameters_handler_->getParamGetter(name_);

    getParam(power_, "cost_power", 1);
    getParam(weight_, "cost_weight", 5.0);
    getParam(threshold_to_consider_, "threshold_to_consider", 1.4);

    RCLCPP_INFO(logger_, "GoalCritic instantiated with %d power and %f weight.", power_, weight_);
}

void GoalCritic::score(CriticData& data)
{
    if (!enabled_ || !utils::withinPositionGoalTolerance(threshold_to_consider_, data.state.pose.pose, data.path))
    {
        return;
    }

    const auto goal_idx = data.path.x.shape(0) - 1;
    const auto goal_x = data.path.x(goal_idx);
    const auto goal_y = data.path.y(goal_idx);

    const auto traj_x = xt::view(data.trajectories.x, xt::all(), xt::all());
    const auto traj_y = xt::view(data.trajectories.y, xt::all(), xt::all());

    size_t num_trajectories = traj_x.shape(0);
    size_t num_points = traj_x.shape(1);

    // Flatten the trajectory arrays for CUDA
    auto flat_traj_x = xt::flatten(traj_x);
    auto flat_traj_y = xt::flatten(traj_y);

    // Allocate memory on the GPU with RAII wrapper
    std::unique_ptr<float, decltype(&cudaFree)> d_traj_x(nullptr, &cudaFree);
    std::unique_ptr<float, decltype(&cudaFree)> d_traj_y(nullptr, &cudaFree);
    std::unique_ptr<float, decltype(&cudaFree)> d_dists(nullptr, &cudaFree);

    cudaError_t err;

    float* raw_d_traj_x = nullptr;
    err = cudaMalloc(&raw_d_traj_x, flat_traj_x.size() * sizeof(float));
    if (err != cudaSuccess) {
        RCLCPP_ERROR(logger_, "CUDA memory allocation failed for d_traj_x: %s", cudaGetErrorString(err));
        return;
    }
    d_traj_x.reset(raw_d_traj_x);

    float* raw_d_traj_y = nullptr;
    err = cudaMalloc(&raw_d_traj_y, flat_traj_y.size() * sizeof(float));
    if (err != cudaSuccess) {
        RCLCPP_ERROR(logger_, "CUDA memory allocation failed for d_traj_y: %s", cudaGetErrorString(err));
        return;
    }
    d_traj_y.reset(raw_d_traj_y);

    float* raw_d_dists = nullptr;
    err = cudaMalloc(&raw_d_dists, flat_traj_x.size() * sizeof(float));
    if (err != cudaSuccess) {
        RCLCPP_ERROR(logger_, "CUDA memory allocation failed for d_dists: %s", cudaGetErrorString(err));
        return;
    }
    d_dists.reset(raw_d_dists);

    // Copy data to the GPU
    err = cudaMemcpy(d_traj_x.get(), flat_traj_x.data(), flat_traj_x.size() * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        RCLCPP_ERROR(logger_, "CUDA memory copy failed for d_traj_x: %s", cudaGetErrorString(err));
        return;
    }

    err = cudaMemcpy(d_traj_y.get(), flat_traj_y.data(), flat_traj_y.size() * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        RCLCPP_ERROR(logger_, "CUDA memory copy failed for d_traj_y: %s", cudaGetErrorString(err));
        return;
    }

    // Launch the kernel
    int threads_per_block = 256;
    int blocks = (flat_traj_x.size() + threads_per_block - 1) / threads_per_block;
    computeDistancesKernel<<<blocks, threads_per_block>>>(d_traj_x.get(), d_traj_y.get(), goal_x, goal_y, d_dists.get(),
                                                          num_trajectories, num_points);

    // Check for kernel launch errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        RCLCPP_ERROR(logger_, "CUDA kernel launch failed: %s", cudaGetErrorString(err));
        return;
    }

    // Synchronize the device
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        RCLCPP_ERROR(logger_, "CUDA device synchronization failed: %s", cudaGetErrorString(err));
        return;
    }

    // Retrieve results from the GPU
    std::vector<float> h_dists(flat_traj_x.size());
    err = cudaMemcpy(h_dists.data(), d_dists.get(), flat_traj_x.size() * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        RCLCPP_ERROR(logger_, "CUDA memory copy failed for d_dists: %s", cudaGetErrorString(err));
        return;
    }

    // Adapt GPU results to xtensor and compute costs
    xt::xarray<float> dists = xt::adapt(h_dists, {num_trajectories, num_points});
    data.costs += xt::pow(xt::mean(dists, {1}, immediate) * weight_, power_);
}

} // namespace mppi::critics

#include <pluginlib/class_list_macros.hpp>
PLUGINLIB_EXPORT_CLASS(mppi::critics::GoalCritic, mppi::critics::CriticFunction)
