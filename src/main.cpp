#include <iostream>
#include <pcg_random.hpp>
#include <nlohmann/json.hpp>
#include "train_track.hpp"

int main()
{
    pcg32 rng(pcg_extras::seed_seq_from<std::random_device>{});

    TrainTrack tt;
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
                    if (intersections >= 8)
                    {
                        std::cout << nlohmann::json(tt).dump() << std::endl;
                        std::cout << "(" << intersections << " " << nlohmann::json(configuration) << ")" << std::endl;
                        return 0;
                    }
                }
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
