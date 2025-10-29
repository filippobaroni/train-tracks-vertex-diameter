#include <atomic>
#include <iostream>
#include <mutex>
#include <thread>
#include <pcg_random.hpp>
#include <nlohmann/json.hpp>
#include "train_track.hpp"
#include <csignal>

std::atomic<bool> running = true;

void signal_handler(int)
{
    running = false;
}

std::mutex mutex;
unsigned max_intersections = 0;
unsigned long global_attempts = 0;

void worker()
{
    pcg32 rng(pcg_extras::seed_seq_from<std::random_device>{});

    TrainTrack my_best_tt;
    unsigned my_max_intersections = 0;
    CarriedCurvesConfiguration my_best_configuration;
    unsigned attempts = 0;
    while (running)
    {
        auto tt = TrainTrack::random_trivalent_train_track(rng, 12);
        auto [g, p] = tt.get_surface();
        if (g == 3 && p == 0)
        {
            const auto vertex_measures = tt.get_vertex_measures<int>();
            for (size_t i = 0; i < vertex_measures.size(); ++i)
            {
                for (size_t j = i + 1; j < vertex_measures.size(); ++j)
                {
                    const auto [intersections, configuration] = tt.compute_intersection_number(
                        vertex_measures[i], vertex_measures[j]);
                    if (intersections > my_max_intersections)
                    {
                        my_max_intersections = intersections;
                        my_best_tt = tt;
                        my_best_configuration = configuration;
                    }
                }
            }
            ++attempts;
            if (attempts % 1000 == 0)
            {
                std::lock_guard lock(mutex);
                if (my_max_intersections > max_intersections)
                {
                    max_intersections = my_max_intersections;
                    std::cout << "New best intersection number found: " << max_intersections << std::endl
                        << nlohmann::json(my_best_tt) << std::endl
                        << nlohmann::json(my_best_configuration) << std::endl;
                }
                global_attempts += attempts;
                std::cout << "Total attempts: " << global_attempts << std::endl;
                attempts = 0;
            }
        }
    }
}

int main()
{
    std::signal(SIGINT, signal_handler);

    const auto num_threads = std::max(1U, std::thread::hardware_concurrency());
    std::vector<std::thread> threads;
    threads.reserve(num_threads);
    for (unsigned i = 0; i < num_threads; ++i)
    {
        threads.emplace_back(worker);
    }

    for (auto& t : threads)
    {
        t.join();
    }

    return 0;
}
