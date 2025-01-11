## Benchmark dir:
`nav2_mppi_controller/benchmark/gpu_controller_benchmark.cpp`

This is the file that will run the controller

## GPU optimization

In each critic in `critics` dir, there is a `score()` method. It scores all the trajectories in a loop. We want to score all those trajectories at once using GPU.