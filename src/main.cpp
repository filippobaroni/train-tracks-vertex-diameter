#include <iostream>
#include <pcg_random.hpp>
#include <nlohmann/json.hpp>
#include "train_track.hpp"

int main()
{
    pcg32 rng(pcg_extras::seed_seq_from<std::random_device>{});

    TrainTrack tt;
    unsigned max_intersections = 0;
    unsigned attempts = 0;
    while (true)
    {
        tt = TrainTrack::random_trivalent_train_track(rng, 8);
        auto [g, p] = tt.get_surface();
        if (g == 2 && p == 0)
        {
            const auto vertex_measures = tt.get_vertex_measures<int>();
            for (size_t i = 0; i < vertex_measures.size(); ++i)
            {
                for (size_t j = i + 1; j < vertex_measures.size(); ++j)
                {
                    const auto [intersections, configuration] = tt.compute_intersection_number(vertex_measures[i], vertex_measures[j]);
                    if (intersections > max_intersections)
                    {
                        max_intersections = intersections;
                        std::cout << "New max intersections: " << max_intersections << std::endl
                                  << nlohmann::json(tt).dump() << std::endl
                                  << "Measures: " << nlohmann::json(vertex_measures[i]).dump() << " , " << nlohmann::json(vertex_measures[j]).dump() << std::endl
                                  << "Configuration: " << nlohmann::json(configuration).dump() << std::endl;
                    }
                    // const auto initial_configuration = configuration_from_carried_curves(vertex_measures[i], vertex_measures[j]);
                    // const auto initial_intersections = intersections_in_configuration(tt, initial_configuration);

                    // const auto [intersections, configuration] = tt.compute_intersection_number(vertex_measures[i], vertex_measures[j]);
                    // if (intersections + 6 <= initial_intersections)
                    // {
                    //     std::cout << nlohmann::json(tt).dump() << std::endl;
                    //     std::cout << "(" << initial_intersections << " " << nlohmann::json(initial_configuration) << ")" << std::endl;
                    //     std::cout << "-> (" << intersections << " " << nlohmann::json(configuration) << ")" << std::endl;
                    //     return 0;
                    // }
                }
            }
            ++attempts;
            if (attempts % 1000 == 0)
            {
                std::cout << "Attempts: " << attempts << ", Current max intersections: " << max_intersections << std::endl;
            }
        }
    }
    // TrainTrack tt(1, 6);
    // tt.attach_branch(0, 0, LeftRight::Left);
    // tt.attach_branch(0, 0, LeftRight::Left);
    // tt.attach_branch(1, 0, LeftRight::Right);
    // tt.attach_branch(1, 0, LeftRight::Right);
    // tt.attach_branch(2, 0, LeftRight::Left);
    // tt.attach_branch(2, 0, LeftRight::Left);
    // tt.attach_branch(3, 0, LeftRight::Right);
    // tt.attach_branch(3, 0, LeftRight::Right);
    // tt.attach_branch(4, 0, LeftRight::Left);
    // tt.attach_branch(4, 0, LeftRight::Left);
    // tt.attach_branch(5, 0, LeftRight::Right);
    // tt.attach_branch(5, 0, LeftRight::Right);
    // if (!tt.finalize())
    // {
    //     std::cerr << "Failed to finalize train track" << std::endl;
    //     return 1;
    // }
    // std::cout << nlohmann::json(tt).dump() << std::endl;
    tt.get_vertex_measures<int>();

    return 0;
}
