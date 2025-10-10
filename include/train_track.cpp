#include "train_track.hpp"
#include <stdexcept>
#include <ranges>
#include <algorithm>
#include <boost/lambda2.hpp>
#include <numeric>
#include <random>

using namespace std::ranges;
using namespace boost::lambda2;

template <typename E>
constexpr E flip(E e)
{
    return static_cast<E>(1 - static_cast<int>(e));
}

size_t TrainTrack::add_switch()
{
    if (is_finalized)
    {
        throw std::runtime_error("Cannot add switch to finalized TrainTrack");
    }
    switches.emplace_back();
    return switches.size() - 1;
}

size_t TrainTrack::add_branch()
{
    if (is_finalized)
    {
        throw std::runtime_error("Cannot add branch to finalized TrainTrack");
    }
    branches.emplace_back();
    return branches.size() - 1;
}

void TrainTrack::attach_branch(size_t branch_index, size_t switch_index, LeftRight side)
{
    attach_branch(branch_index, switch_index, side, switches[switch_index].connections[static_cast<int>(side)].size());
}

void TrainTrack::attach_branch(size_t branch_index, size_t switch_index, LeftRight side, size_t pos)
{
    if (is_finalized)
    {
        throw std::runtime_error("Cannot attach branch to finalized TrainTrack");
    }
    Branch &branch = branches[branch_index];
    if (branch.endpoints.size() >= 2)
    {
        throw std::runtime_error("Branch already has two endpoints");
    }
    branch.endpoints.push_back({switch_index, side, 0, (side == LeftRight::Left) == (branch.endpoints.size() == 1) ? UpDown::Up : UpDown::Down});
    auto &conns = switches[switch_index].connections[static_cast<int>(side)];
    conns.insert(conns.begin() + pos, {branch_index, branch.endpoints.size() == 1 ? FirstSecond::First : FirstSecond::Second});
}

TrainTrack::Surface TrainTrack::get_surface()
{
    if (!is_finalized)
    {
        throw std::runtime_error("Cannot get surface of non-finalized TrainTrack");
    }
    return surface;
}

bool TrainTrack::finalize(const TrainTrackOptions &options)
{
    if (is_finalized)
    {
        // Already finalised
        return true;
    }
    if (any_of(branches, [](const Branch &b)
               { return b.endpoints.size() != 2; }))
    {
        // A branch with less than two endpoints
        return false;
    }
    if (any_of(switches, [](const Switch &sw)
               { return sw.connections[static_cast<int>(LeftRight::Left)].empty() || sw.connections[static_cast<int>(LeftRight::Right)].empty(); }))
    {
        // A switch with no incident branches on one side
        return false;
    }

    // Set the position of each endpoint in the corresponding switch
    for (auto &sw : switches)
    {
        for (auto &conn : sw.connections)
        {
            for (size_t i = 0; i < conn.size(); ++i)
            {
                branches[conn[i].branch].endpoints[static_cast<int>(conn[i].endpoint)].position = i;
            }
        }
    }

    // Compute complementary regions
    compute_complementary_regions(options);

    // Compute surface
    const int euler_characteristic = int(switches.size()) - int(branches.size()) + int(complementary_regions.size());
    if (euler_characteristic % 2 != 0)
    {
        throw std::runtime_error("Euler characteristic is not even");
    }
    surface.genus = (2 - euler_characteristic) / 2;
    surface.punctures = std::accumulate(complementary_regions.begin(), complementary_regions.end(), 0u, [](unsigned int sum, const ComplementaryRegion &cr)
                                        { return sum + cr.punctures; });

    is_finalized = true;
    return true;
}

void TrainTrack::compute_complementary_regions(const TrainTrackOptions &options)
{
    std::array<std::vector<bool>, 2> visited;
    visited.fill(std::vector(branches.size(), false));

    for (const auto side_of_branch0 : {UpDown::Up, UpDown::Down})
    {
        for (size_t b0 = 0; b0 < branches.size(); ++b0)
        {
            if (visited[static_cast<int>(side_of_branch0)][b0])
            {
                continue;
            }

            unsigned int cusps = 0;

            size_t b = b0;
            auto side_of_branch = side_of_branch0;
            auto endpoint_index = FirstSecond::First;
            do
            {
                visited[static_cast<int>(side_of_branch)][b] = true;
                const auto [sw, sw_side, position, orientation] = branches[b].endpoints[static_cast<int>(flip(endpoint_index))];
                const int next_position = int(position) + (side_of_branch == orientation ? -1 : 1);
                const int max_position = switches[sw].connections[static_cast<int>(sw_side)].size();
                if (next_position == -1)
                {
                    const auto &germ = switches[sw].connections[static_cast<int>(flip(sw_side))].front();
                    b = germ.branch;
                    endpoint_index = germ.endpoint;
                    side_of_branch = branches[b].endpoints[static_cast<int>(endpoint_index)].orientation;
                }
                else if (next_position == max_position)
                {
                    const auto &germ = switches[sw].connections[static_cast<int>(flip(sw_side))].back();
                    b = germ.branch;
                    endpoint_index = germ.endpoint;
                    side_of_branch = flip(branches[b].endpoints[static_cast<int>(endpoint_index)].orientation);
                }
                else
                {
                    const auto &germ = switches[sw].connections[static_cast<int>(sw_side)][next_position];
                    b = germ.branch;
                    endpoint_index = germ.endpoint;
                    side_of_branch = orientation == branches[b].endpoints[static_cast<int>(endpoint_index)].orientation ? flip(side_of_branch) : side_of_branch;
                    ++cusps;
                }
            } while (!visited[static_cast<int>(side_of_branch)][b]);
            unsigned int punctures = 0;
            if (options.add_punctures)
            {
                if (cusps == 0)
                {
                    punctures = 2;
                }
                else if (cusps == 1)
                {
                    punctures = 1;
                }
            }
            complementary_regions.push_back({cusps, punctures});
        }
    }
}

template <typename URBG>
TrainTrack TrainTrack::random_trivalent_train_track(URBG &rng, size_t switches_count, const TrainTrackOptions &options)
{
    if (switches_count % 2 != 0)
    {
        throw std::runtime_error("Number of switches must be even for a trivalent train track");
    }

    std::uniform_int_distribution<int> coin_flip(0, 1);

    size_t branches_count = 3 * switches_count / 2;
    TrainTrack tt(switches_count, branches_count);

    std::vector<std::tuple<size_t, LeftRight>> attachments;
    attachments.reserve(3 * switches_count);
    for (size_t s = 0; s < switches_count; ++s)
    {
        attachments.emplace_back(s, LeftRight::Left);
        attachments.emplace_back(s, LeftRight::Right);
        attachments.emplace_back(s, static_cast<LeftRight>(coin_flip(rng)));
    }
    shuffle(attachments, rng);

    std::vector<size_t> branches;
    branches.reserve(2 * branches_count);
    for (size_t b = 0; b < branches_count; ++b)
    {
        branches.push_back(b);
        branches.push_back(b);
    }
    shuffle(branches, rng);

    for (const auto [b, att] : views::zip(branches, attachments))
    {
        tt.attach_branch(b, get<0>(att), get<1>(att));
    }

    tt.finalize(options);
    return tt;
}

void to_json(nlohmann::json &j, const TrainTrack &tt)
{
    if (!tt.is_finalized)
    {
        throw std::runtime_error("Cannot serialize non-finalized TrainTrack");
    }
    j = nlohmann::json{
        {"switches", tt.switches},
        {"branches", tt.branches},
        {"complementary_regions", tt.complementary_regions},
        {"surface", tt.surface}};
}

void from_json(const nlohmann::json &j, TrainTrack &tt)
{
    j.at("switches").get_to(tt.switches);
    j.at("branches").get_to(tt.branches);
    j.at("complementary_regions").get_to(tt.complementary_regions);
    j.at("surface").get_to(tt.surface);
    tt.is_finalized = true;
}