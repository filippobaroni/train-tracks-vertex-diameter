#include <iostream>
#include <pcg_random.hpp>
#include <nlohmann/json.hpp>
#include "train_track.hpp"

int main()
{
    pcg32 rng(pcg_extras::seed_seq_from<std::random_device>{});

    TrainTrack tt;
    unsigned int attempts = 0;
    while (true)
    {
        tt = TrainTrack::random_trivalent_train_track(rng, 8);
        auto [g, p] = tt.get_surface();
        if (g == 2 && p == 0)
        {
            tt.get_vertex_measures<int>();
        }
    }
    // TrainTrack tt(1, 6);
    // tt.attach_branch(0, 0, TrainTrack::LeftRight::Left);
    // tt.attach_branch(0, 0, TrainTrack::LeftRight::Left);
    // tt.attach_branch(1, 0, TrainTrack::LeftRight::Right);
    // tt.attach_branch(1, 0, TrainTrack::LeftRight::Right);
    // tt.attach_branch(2, 0, TrainTrack::LeftRight::Left);
    // tt.attach_branch(2, 0, TrainTrack::LeftRight::Left);
    // tt.attach_branch(3, 0, TrainTrack::LeftRight::Right);
    // tt.attach_branch(3, 0, TrainTrack::LeftRight::Right);
    // tt.attach_branch(4, 0, TrainTrack::LeftRight::Left);
    // tt.attach_branch(4, 0, TrainTrack::LeftRight::Left);
    // tt.attach_branch(5, 0, TrainTrack::LeftRight::Right);
    // tt.attach_branch(5, 0, TrainTrack::LeftRight::Right);
    // if (!tt.finalize())
    // {
    //     std::cerr << "Failed to finalize train track" << std::endl;
    //     return 1;
    // }
    // tt.get_vertex_measures<int>();

    return 0;
}
