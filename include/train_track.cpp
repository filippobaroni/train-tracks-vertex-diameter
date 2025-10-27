#include "train_track.hpp"
#include <stdexcept>
#include <ranges>
#include <algorithm>
#include <boost/lambda2.hpp>
#include <boost/dynamic_bitset.hpp>
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

inline const auto &TrainTrack::get_surface() const
{
    return surface;
}

inline const auto &TrainTrack::get_switches() const
{
    return switches;
}

inline const auto &TrainTrack::get_branches() const
{
    return branches;
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

    // Check connectedness
    {
        std::vector<bool> visited_switches(switches.size(), false);
        const auto dfs = [&](const auto &self, size_t sw) -> void
        {
            visited_switches[sw] = true;
            for (const auto side : {LeftRight::Left, LeftRight::Right})
            {
                for (const auto &germ : switches[sw].connections[static_cast<int>(side)])
                {
                    const auto &branch = branches[germ.branch];
                    const auto &other_endpoint = branch.endpoints[static_cast<int>(flip(germ.endpoint))];
                    if (!visited_switches[other_endpoint.sw])
                    {
                        self(self, other_endpoint.sw);
                    }
                }
            }
        };
        dfs(dfs, 0);
        if (!all_of(visited_switches, _1))
        {
            // Not all switches are connected
            return false;
        }
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
            std::vector<ComplementaryRegion::BranchSide> boundary_sequence;
            do
            {
                boundary_sequence.push_back({b, side_of_branch});
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
            complementary_regions.push_back({cusps, punctures, boundary_sequence});
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

    // Declare all fmpz and fmpz_mat variables
    fmpz_t int_denominator,
        int_tmp;
    fmpz_mat_t mat_switch_equations,
        mat_nullspace_basis_extended,
        mat_nullspace_basis,
        mat_nullspace_basis_rre,
        mat_nullspace_change_of_basis;

    // Matrix for switch equations
    fmpz_mat_init(mat_switch_equations, switches.size(), branches.size());
    fmpz_mat_zero(mat_switch_equations);
    for (size_t sw = 0; sw < switches.size(); ++sw)
    {
        const auto &swc = switches[sw].connections;
        for (const auto &germ : swc[static_cast<int>(LeftRight::Left)])
        {
            fmpz_set_si(fmpz_mat_entry(mat_switch_equations, sw, germ.branch), 1);
        }
        for (const auto &germ : swc[static_cast<int>(LeftRight::Right)])
        {
            fmpz_set_si(fmpz_mat_entry(mat_switch_equations, sw, germ.branch), -1);
        }
    }
    // std::cout << "Switch equations matrix:" << std::endl;
    // fmpz_mat_print_pretty(mat_switch_equations);
    // std::cout << std::endl;

    // Matrix for nullspace
    fmpz_mat_init(mat_nullspace_basis_extended, branches.size(), branches.size());
    const auto nullspace_dim = fmpz_mat_nullspace(mat_nullspace_basis_extended, mat_switch_equations);
    fmpz_mat_window_init(mat_nullspace_basis, mat_nullspace_basis_extended, 0, 0, branches.size(), nullspace_dim);
    // std::cout << "Nullspace basis matrix:" << std::endl;
    // fmpz_mat_print_pretty(mat_nullspace_basis);
    // std::cout << std::endl;

    // Find linearly independent rows
    std::vector<slong> permutation(branches.size());
    std::iota(permutation.begin(), permutation.end(), 0);
    fmpz_mat_init(mat_nullspace_basis_rre, branches.size(), nullspace_dim);
    fmpz_mat_fflu(mat_nullspace_basis_rre, int_denominator, permutation.data(), mat_nullspace_basis, 0);
    // std::cout << "mat_nullspace_basis_rre of nullspace basis matrix:" << std::endl;
    // fmpz_mat_print_pretty(mat_nullspace_basis_rre);
    // std::cout << std::endl
    //           << "Permutation: ";
    // for (const auto p : permutation)
    // {
    //     std::cout << p << " ";
    // }
    // std::cout << std::endl;

    // Compute projection to kernel
    fmpz_mat_init(mat_nullspace_change_of_basis, nullspace_dim, nullspace_dim);
    for (slong i = 0; i < nullspace_dim; ++i)
    {
        for (slong j = 0; j < nullspace_dim; ++j)
        {
            fmpz_set(fmpz_mat_entry(mat_nullspace_change_of_basis, i, j), fmpz_mat_entry(mat_nullspace_basis, permutation[i], j));
        }
    }
    // std::cout << "mat_nullspace_change_of_basis matrix:" << std::endl;
    // fmpz_mat_print_pretty(mat_nullspace_change_of_basis);
    // std::cout << std::endl;
    fmpz_mat_inv(mat_nullspace_change_of_basis, int_denominator, mat_nullspace_change_of_basis);
    fmpz_mat_mul(mat_nullspace_basis, mat_nullspace_basis, mat_nullspace_change_of_basis);
    fmpz_mat_content(int_tmp, mat_nullspace_basis);
    if ((fmpz_cmp_ui(int_denominator, 0) < 0) != (fmpz_cmp_ui(int_tmp, 0) < 0))
    {
        fmpz_neg(int_tmp, int_tmp);
    }
    fmpz_mat_scalar_divexact_fmpz(mat_nullspace_basis, mat_nullspace_basis, int_tmp);
    fmpz_divexact(int_denominator, int_denominator, int_tmp);

    // std::cout << "Projection to kernel matrix:" << std::endl;
    // fmpz_mat_print_pretty(mat_nullspace_basis);
    // std::cout << std::endl
    //           << "Denominator: ";
    // fmpz_print(int_denominator);
    // std::cout << std::endl;

    // All candidate measures
    std::array<std::vector<Measure<T>>, 3> basis_measures;
    T basis_measures_denominator = static_cast<T>(fmpz_get_si(int_denominator));
    if (basis_measures_denominator > 2 || basis_measures_denominator < 1)
    {
        // It seems like the denominator is always 1 or 2, but I'm not sure why
        throw std::runtime_error("Denominator is greater than 2??");
    }
    for (slong i = 0; i < nullspace_dim; ++i)
    {
        basis_measures[1].emplace_back(branches.size());
        basis_measures[2].emplace_back(branches.size());
        for (slong j = 0; j < static_cast<slong>(branches.size()); ++j)
        {
            basis_measures[1][i][j] = static_cast<T>(fmpz_get_si(fmpz_mat_entry(mat_nullspace_basis, j, i)));
            basis_measures[2][i][j] = 2 * basis_measures[1][i][j];
        }
    }

    std::vector<Measure<T>> candidate_measures;
    std::vector<boost::dynamic_bitset<>> supports;
    auto recursive = [&](auto self, Measure<T> &current, unsigned int index) -> void
    {
        for (int i = 0; i <= 2; ++i)
        {
            if (index + 1 == nullspace_dim)
            {

                if (
                    all_of(current,
                           _1 >= 0 &&
                               _1 <= 2 * basis_measures_denominator &&
                               !(_1 & (basis_measures_denominator - 1))) &&
                    any_of(current, _1 == basis_measures_denominator))
                {
                    candidate_measures.emplace_back(current | views::transform(_1 >> (basis_measures_denominator - 1)) | to<Measure<T>>());
                    supports.emplace_back(current | views::transform(_1 != 0) | to<boost::dynamic_bitset<>>());
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

    // Extract measures of minimal support (i.e. vertex measures)
    sort(supports);
    std::set<boost::dynamic_bitset<>> vertex_supports;
    for (const auto &s : supports)
    {
        if (none_of(vertex_supports, [&](const auto &vs)
                    { return vs.is_subset_of(s); }))
        {
            vertex_supports.insert(s);
        }
    }
    const auto vertex_measures = candidate_measures | views::filter([&](const auto &m)
                                                                    { return vertex_supports.contains(m | views::transform(_1 != 0) | to<boost::dynamic_bitset<>>()); }) |
                                 to<std::vector<Measure<T>>>();

    // Deallocate all the matrices
    fmpz_clear(int_denominator);
    fmpz_clear(int_tmp);
    fmpz_mat_clear(mat_switch_equations);
    fmpz_mat_clear(mat_nullspace_basis_extended);
    fmpz_mat_window_clear(mat_nullspace_basis);
    fmpz_mat_clear(mat_nullspace_basis_rre);
    fmpz_mat_clear(mat_nullspace_change_of_basis);

    // Return vertex measures
    return vertex_measures;
}

template <std::integral T>
CarriedCurvesConfiguration configuration_from_carried_curves(const Measure<T> &m1, const Measure<T> &m2)
{
    CarriedCurvesConfiguration config;
    for (size_t i = 0; i < m1.size(); ++i)
    {
        std::vector<FirstSecond> v(m1[i] + m2[i], FirstSecond::Second);
        for (size_t j = 0; static_cast<T>(j) < m1[i]; ++j)
        {
            v[j] = FirstSecond::First;
        }
        config.emplace_back(std::move(v));
    }
    return config;
}

unsigned long intersections_in_configuration(const TrainTrack &train_track, const CarriedCurvesConfiguration &config)
{
    unsigned long intersections = 0;
    for (size_t sw = 0; sw < train_track.get_switches().size(); ++sw)
    {
        const auto left_range = ConfigurationSwitchEndpointsRange(train_track, config, sw, LeftRight::Left);
        const auto right_range = ConfigurationSwitchEndpointsRange(train_track, config, sw, LeftRight::Right);
        auto left_it = left_range.begin(), right_it = right_range.begin();
        int left_count = 0, right_count = 0;
        while (true)
        {
            while (left_it != left_range.end() && config[train_track.get_switches()[sw].connections[static_cast<int>(LeftRight::Left)][left_it->connection_position].branch][left_it->position_on_branch] == FirstSecond::Second)
            {
                ++left_it;
                ++left_count;
            }
            if (left_it == left_range.end())
            {
                break;
            }
            while (config[train_track.get_switches()[sw].connections[static_cast<int>(LeftRight::Right)][right_it->connection_position].branch][right_it->position_on_branch] == FirstSecond::Second)
            {
                ++right_it;
                ++right_count;
            }
            intersections += std::abs(left_count - right_count);
            ++left_it;
            ++right_it;
        }
    }
    return intersections;
}

template <std::integral T>
std::tuple<unsigned long, CarriedCurvesConfiguration> TrainTrack::compute_intersection_number(const Measure<T> &m1, const Measure<T> &m2) const
{
    if (!is_finalized)
    {
        throw std::runtime_error("Cannot compute intersection number on non-finalized TrainTrack");
    }
    auto c = configuration_from_carried_curves(m1, m2);
    unsigned long intersections = intersections_in_configuration(*this, c);
    return {intersections, c};
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

    while (true)
    {
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

        if (tt.finalize(options))
        {
            return tt;
        }
    }
}

ConfigurationSwitchEndpointsRange::ConfigurationSwitchEndpointsRange(const TrainTrack &tt, const CarriedCurvesConfiguration &config, size_t sw_index, LeftRight side)
    : train_track(tt), configuration(config), switch_index(sw_index), side(side)
{
}

ConfigurationSwitchEndpointsRange::Iterator::Iterator(const TrainTrack &train_track, const CarriedCurvesConfiguration &configuration, size_t switch_index, LeftRight side, size_t connection_position, int position_on_branch)
    : train_track(train_track), configuration(configuration), switch_index(switch_index), side(side), value{connection_position, position_on_branch}
{
}

auto ConfigurationSwitchEndpointsRange::Iterator::operator*() const -> reference
{
    return value;
}

auto ConfigurationSwitchEndpointsRange::Iterator::operator->() const -> pointer
{
    return &value;
}

bool ConfigurationSwitchEndpointsRange::Iterator::operator==(const Iterator &other) const
{
    return value.connection_position == other.value.connection_position &&
           value.position_on_branch == other.value.position_on_branch;
}

bool ConfigurationSwitchEndpointsRange::Iterator::operator!=(const Iterator &other) const
{
    return !(*this == other);
}

auto ConfigurationSwitchEndpointsRange::Iterator::skip_empty_branches() -> Iterator &
{
    const auto &conns = train_track.get_switches()[switch_index].connections[static_cast<int>(side)];
    while (value.connection_position < conns.size() && configuration[conns[value.connection_position].branch].empty())
    {
        ++value.connection_position;
    }
    if (value.connection_position < conns.size())
    {
        const auto &c = conns[value.connection_position];
        value.position_on_branch = train_track.get_branches()[c.branch].endpoints[static_cast<int>(c.endpoint)].orientation == UpDown::Up ? 0 : static_cast<int>(configuration[c.branch].size() - 1);
    }
    else
    {
        value.position_on_branch = 0;
    }
    return *this;
}

auto ConfigurationSwitchEndpointsRange::Iterator::operator++() -> Iterator &
{
    const auto &conns = train_track.get_switches()[switch_index].connections[static_cast<int>(side)];
    const auto &c = conns[value.connection_position];
    if (train_track.get_branches()[c.branch].endpoints[static_cast<int>(c.endpoint)].orientation == UpDown::Up)
    {
        if (static_cast<size_t>(++value.position_on_branch) >= configuration[c.branch].size())
        {
            ++value.connection_position;
            return skip_empty_branches();
        }
    }
    else
    {
        if (value.position_on_branch-- <= 0)
        {
            ++value.connection_position;
            return skip_empty_branches();
        }
    }
    return *this;
}

auto ConfigurationSwitchEndpointsRange::Iterator::operator++(int) -> Iterator
{
    Iterator tmp = *this;
    ++(*this);
    return tmp;
}

auto ConfigurationSwitchEndpointsRange::begin() const -> Iterator
{
    const auto &conns = train_track.get_switches()[switch_index].connections[static_cast<int>(side)];
    auto it = Iterator(train_track, configuration, switch_index, side, 0, 0);
    it.skip_empty_branches();
    if (it->connection_position >= conns.size())
    {
        return Iterator(train_track, configuration, switch_index, side, conns.size(), 0);
    }
    const auto &c = conns[it->connection_position];
    const int position_on_branch = train_track.get_branches()[c.branch].endpoints[static_cast<int>(c.endpoint)].orientation == UpDown::Up ? 0 : static_cast<int>(configuration[c.branch].size() - 1);
    return Iterator(train_track, configuration, switch_index, side, it->connection_position, position_on_branch);
}

auto ConfigurationSwitchEndpointsRange::end() const -> Iterator
{
    return Iterator(train_track, configuration, switch_index, side, train_track.get_switches()[switch_index].connections[static_cast<int>(side)].size(), 0);
}

NLOHMANN_JSON_SERIALIZE_ENUM(LeftRight, {{LeftRight::Left, 0}, {LeftRight::Right, 1}});
NLOHMANN_JSON_SERIALIZE_ENUM(UpDown, {{UpDown::Up, "up"}, {UpDown::Down, "down"}});
NLOHMANN_JSON_SERIALIZE_ENUM(FirstSecond, {{FirstSecond::First, 0}, {FirstSecond::Second, 1}});
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(TrainTrack::ComplementaryRegion::BranchSide, branch, side);
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(TrainTrack::ComplementaryRegion, cusps, punctures, boundary);
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(TrainTrack::Surface, genus, punctures);

void to_json(nlohmann::json &j, const TrainTrack::Branch::Endpoint &e)
{
    j = nlohmann::json{
        {"switch", e.sw},
        {"side", e.side},
        {"position", e.position},
        {"orientation", e.orientation}};
}
void from_json(const nlohmann::json &j, TrainTrack::Branch::Endpoint &e)
{
    j.at("switch").get_to(e.sw);
    j.at("side").get_to(e.side);
    j.at("position").get_to(e.position);
    j.at("orientation").get_to(e.orientation);
}

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(TrainTrack::Branch, endpoints);
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(TrainTrack::Switch::Germ, branch, endpoint);
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(TrainTrack::Switch, connections);

void to_json(nlohmann::json &j, const TrainTrack &tt)
{
    if (!tt.is_finalized)
    {
        throw std::runtime_error("Cannot serialize non-finalized TrainTrack");
    }
    j = nlohmann::json{
        {"switches", tt.get_switches()},
        {"branches", tt.get_branches()},
        {"complementaryRegions", tt.complementary_regions},
        {"surface", tt.get_surface()}};
}

void from_json(const nlohmann::json &j, TrainTrack &tt)
{
    j.at("switches").get_to(tt.switches);
    j.at("branches").get_to(tt.branches);
    j.at("complementaryRegions").get_to(tt.complementary_regions);
    j.at("surface").get_to(tt.surface);
    tt.is_finalized = true;
}