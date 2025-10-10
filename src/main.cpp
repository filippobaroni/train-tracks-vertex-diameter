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
        ++attempts;
        tt = TrainTrack::random_trivalent_train_track(rng, 6);
        auto [g, p] = tt.get_surface();
        if (g == 2 && p == 0)
        {
            break;
        }
    }

    std::cout << nlohmann::json(tt).dump(2) << std::endl;
    std::cerr << "Found in " << attempts << " attempts" << std::endl;

    const auto vertex_measures = tt.get_vertex_measures<int>();

    return 0;
}
