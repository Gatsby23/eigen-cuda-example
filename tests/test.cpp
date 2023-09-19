#include "utils.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <example.h>

#include <iostream>

TEST_CASE("Test Point-Point Distances", "[point-point]")
{
    const int N = GENERATE(10, 100'0000);
    std::cout << "We process " << N << " dimension data." << std::endl;
    const int iters = 100;
    double cpu_duration = 0.0;
    double gpu_with_eign_duration = 0.0;
    double gpu_without_eign_duration = 0.0;
    for (int i = 0; i < iters; ++i) {
        const Eigen::MatrixXf V = Eigen::MatrixXf::Random(N, 3);

        std::vector<std::array<int, 2>> point_pairs(N);
        for (auto& pair : point_pairs) {
            for (auto& i : pair) {
                i = std::rand() % N;
            }
        }
        auto cpu_start = std::chrono::steady_clock::now();
        const Eigen::VectorXf r_cpu =
            ece::compute_point_point_distances_cpu(V, point_pairs);
        auto cpu_stop = std::chrono::steady_clock::now();
        double cpu_duration_iterator = std::chrono::duration<double, std::milli>(cpu_stop - cpu_start).count();
        cpu_duration += cpu_duration_iterator;
        auto gpu_start = std::chrono::steady_clock::now();
        const Eigen::VectorXf r_gpu =
            ece::compute_point_point_distances_gpu</*USE_EIGEN=*/true>(
                V, point_pairs);
        auto gpu_stop = std::chrono::steady_clock::now();
        double gpu_duration_iterator = std::chrono::duration<double, std::milli>(gpu_stop - gpu_start).count();
        gpu_with_eign_duration += gpu_duration_iterator;

        auto gpu_no_eigen_start = std::chrono::steady_clock::now();
        const Eigen::VectorXf r_gpu_no_eigen =
            ece::compute_point_point_distances_gpu</*USE_EIGEN=*/false>(
                V, point_pairs);
        auto gpu_no_eigen_stop = std::chrono::steady_clock::now();
        double gpu_no_eigen_duration_iterator = std::chrono::duration<double, std::milli>(gpu_no_eigen_stop - gpu_no_eigen_start).count();
        gpu_without_eign_duration += gpu_no_eigen_duration_iterator;

        if (N <= 10
            && (!r_cpu.isApprox(r_gpu) || !r_cpu.isApprox(r_gpu_no_eigen))) {
            std::cout << "r_cpu         : " << r_cpu.transpose() << std::endl;
            std::cout << "r_gpu         : " << r_gpu.transpose() << std::endl;
            std::cout << "r_gpu_no_eigen: " << r_gpu_no_eigen.transpose()
                      << std::endl;
        }

        CHECK(r_cpu.isApprox(r_gpu));
        // CHECK(r_cpu.isApprox(r_gpu_no_eigen));
    }

    std::cout << "It costs " << cpu_duration << " ms on the cpu." << std::endl;
    std::cout << "It costs " << gpu_with_eign_duration << " ms on the GPU with eigen data structure." << std::endl;
    std::cout << "It costs " << gpu_without_eign_duration << " ms on the GPU without eigen data structure." << std::endl;
}

TEST_CASE("Test Line-Line Distances", "[line-line]")
{
    const int N = 20000 * 24;
    const int iters = 100;

    std::cout << "We process " << N << " dimension data." << std::endl;
    double cpu_duration = 0.0;
    double gpu_with_eign_duration = 0.0;
    double gpu_without_eign_duration = 0.0;

    for (int i = 0; i < iters; ++i) {
        const Eigen::MatrixXf V = Eigen::MatrixXf::Random(N, 3);
        const Eigen::MatrixXi E = random_edges(N, N);
        REQUIRE(E.minCoeff() >= 0);
        REQUIRE(E.maxCoeff() < N);

        std::vector<std::array<int, 2>> line_pairs(N);
        for (auto& pair : line_pairs) {
            pair[0] = std::rand() % N;
            while ((pair[1] = std::rand() % N) == pair[0]) { }
        }

        auto cpu_start = std::chrono::steady_clock::now();
        const Eigen::VectorXf r_cpu =
            ece::compute_line_line_distances_cpu(V, E, line_pairs);
        auto cpu_stop = std::chrono::steady_clock::now();
        double cpu_duration_iterator = std::chrono::duration<double, std::milli>(cpu_stop - cpu_start).count();
        cpu_duration += cpu_duration_iterator;

        auto gpu_start = std::chrono::steady_clock::now();
        const Eigen::VectorXf r_gpu =
            ece::compute_line_line_distances_gpu</*USE_EIGEN=*/true>(
                V, E, line_pairs);
        auto gpu_stop = std::chrono::steady_clock::now();
        double gpu_duration_iterator = std::chrono::duration<double, std::milli>(gpu_stop - gpu_start).count();
        gpu_with_eign_duration += gpu_duration_iterator;

        auto gpu_no_eigen_start = std::chrono::steady_clock::now();
        const Eigen::VectorXf r_gpu_no_eigen =
            ece::compute_line_line_distances_gpu</*USE_EIGEN=*/false>(
                V, E, line_pairs);
        auto gpu_no_eigen_stop = std::chrono::steady_clock::now();
        double gpu_no_eigen_duration_iterator = std::chrono::duration<double, std::milli>(gpu_no_eigen_stop - gpu_no_eigen_start).count();
        gpu_without_eign_duration += gpu_no_eigen_duration_iterator;

        if (N <= 10
            && (!r_cpu.isApprox(r_gpu) || !r_cpu.isApprox(r_gpu_no_eigen))) {
            std::cout << "r_cpu         : " << r_cpu.transpose() << std::endl;
            std::cout << "r_gpu         : " << r_gpu.transpose() << std::endl;
            std::cout << "r_gpu_no_eigen: " << r_gpu_no_eigen.transpose()
                      << std::endl;
        }

        CHECK(r_cpu.isApprox(r_gpu));
        CHECK(r_cpu.isApprox(r_gpu_no_eigen));
    }

    std::cout << "It costs " << cpu_duration << " ms on the cpu." << std::endl;
    std::cout << "It costs " << gpu_with_eign_duration << " ms on the GPU with eigen data structure. Faster " << cpu_duration / gpu_with_eign_duration << " more." << std::endl;
    std::cout << "It costs " << gpu_without_eign_duration << " ms on the GPU without eigen data structure. Faster " << cpu_duration / gpu_without_eign_duration << " more." << std::endl;
}