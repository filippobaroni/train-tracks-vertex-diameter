#include "train_track.hpp"
#include <stdexcept>
#include <ranges>
#include <algorithm>
#include <boost/lambda2.hpp>
#include <numeric>
#include <random>
#include <flint/fmpz_mat.h>
#include <flint/fmpz.h>
#include <set>

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
                else if (cusps == 1 || cusps == 2)
                {
                    punctures = 1;
                }
            }
            complementary_regions.push_back({cusps, punctures});
        }
    }
}

template <std::integral T>
std::vector<Measure<T>> TrainTrack::get_vertex_measures() const
{
    if (!is_finalized)
    {
        throw std::runtime_error("Cannot get vertex measures of non-finalized TrainTrack");
    }

    // Matrix for switch equations
    fmpz_mat_t S;
    fmpz_mat_init(S, switches.size(), branches.size());
    fmpz_mat_zero(S);
    for (size_t sw = 0; sw < switches.size(); ++sw)
    {
        const auto &swc = switches[sw].connections;
        for (const auto &germ : swc[static_cast<int>(LeftRight::Left)])
        {
            fmpz_set_si(fmpz_mat_entry(S, sw, germ.branch), 1);
        }
        for (const auto &germ : swc[static_cast<int>(LeftRight::Right)])
        {
            fmpz_set_si(fmpz_mat_entry(S, sw, germ.branch), -1);
        }
    }
    std::cout << "Switch equations matrix:" << std::endl;
    fmpz_mat_print_pretty(S);
    std::cout << std::endl;

    // Matrix for nullspace
    fmpz_mat_t N, W;
    fmpz_mat_init(N, branches.size(), branches.size());
    const auto nullspace_dim = fmpz_mat_nullspace(N, S);
    fmpz_mat_window_init(W, N, 0, 0, branches.size(), nullspace_dim);
    std::cout << "Nullspace basis matrix:" << std::endl;
    fmpz_mat_print_pretty(W);
    std::cout << std::endl;

    // Find linearly independent rows
    fmpz_mat_t RRE;
    fmpz_t denominator;
    std::vector<slong> permutation(branches.size());
    std::iota(permutation.begin(), permutation.end(), 0);
    fmpz_mat_init(RRE, branches.size(), nullspace_dim);
    fmpz_mat_fflu(RRE, denominator, permutation.data(), W, 0);
    std::cout << "RRE of nullspace basis matrix:" << std::endl;
    fmpz_mat_print_pretty(RRE);
    std::cout << std::endl
              << "Permutation: ";
    for (const auto p : permutation)
    {
        std::cout << p << " ";
    }
    std::cout << std::endl;

    // Compute projection to kernel
    fmpz_mat_t A;
    fmpz_mat_init(A, nullspace_dim, nullspace_dim);
    for (slong i = 0; i < nullspace_dim; ++i)
    {
        for (slong j = 0; j < nullspace_dim; ++j)
        {
            fmpz_set(fmpz_mat_entry(A, i, j), fmpz_mat_entry(W, permutation[i], j));
        }
    }
    std::cout << "A matrix:" << std::endl;
    fmpz_mat_print_pretty(A);
    std::cout << std::endl;
    fmpz_mat_inv(A, denominator, A);
    fmpz_mat_mul(W, W, A);
    std::cout << "Projection to kernel matrix:" << std::endl;
    fmpz_mat_print_pretty(W);
    std::cout << std::endl
              << "Denominator: ";
    fmpz_print(denominator);
    std::cout << std::endl;

    // All candidate measures
    std::array<std::vector<Measure<T>>, 3> basis_measures;
    fmpz_t result;
    for (slong i = 0; i < nullspace_dim; ++i)
    {
        basis_measures[1].emplace_back(branches.size());
        basis_measures[2].emplace_back(branches.size());
        for (slong j = 0; j < static_cast<slong>(branches.size()); ++j)
        {
            const auto entry = fmpz_mat_entry(W, j, i);
            if (!fmpz_divides(result, entry, denominator))
            {
                throw std::runtime_error("It turns out this can be non-integer!");
            }
            else
            {
                basis_measures[1][i][j] = static_cast<T>(fmpz_get_si(result));
                basis_measures[2][i][j] = 2 * basis_measures[1][i][j];
            }
        }
    }

    std::set<Measure<T>> candidate_measures;

    auto recursive = [&](auto self, Measure<T> &current, unsigned int index) -> void
    {
        for (int i = 0; i <= 2; ++i)
        {
            if (index + 1 == nullspace_dim)
            {
                if (all_of(current, _1 >= 0 && _1 <= 2) && any_of(current, _1 == 1))
                {
                    candidate_measures.insert(current);
                }
            }
            else
            {
                self(self, current, index + 1);
            }

            if (i == 2)
            {
                for (size_t j = 0; j < branches.size(); ++j)
                {
                    current[j] -= basis_measures[2][index][j];
                }
            }
            else
            {
                for (size_t j = 0; j < branches.size(); ++j)
                {
                    current[j] += basis_measures[1][index][j];
                }
            }
        }
    };
    Measure<T> tmp_measure(branches.size(), 0);
    recursive(recursive, tmp_measure, 0);

    // Deallocate all the matrices
    fmpz_clear(denominator);
    fmpz_clear(result);
    fmpz_mat_clear(S);
    fmpz_mat_clear(N);
    fmpz_mat_clear(W);
    fmpz_mat_clear(RRE);
    fmpz_mat_clear(A);

    return {};
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