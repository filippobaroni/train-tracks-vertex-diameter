#pragma once
#include <vector>
#include <utility>
#include <concepts>
#include <array>
#include <optional>
#include <random>
#include <nlohmann/json.hpp>

struct TrainTrackOptions
{
    bool add_punctures = true;
};

template <std::integral T>
using Measure = std::vector<T>;

class TrainTrack
{
public:
    enum class LeftRight
    {
        Left = 0,
        Right = 1
    };
    enum class UpDown
    {
        Up = 0,
        Down = 1
    };
    enum class FirstSecond
    {
        First = 0,
        Second = 1
    };
    struct Switch;
    struct Branch;

    struct Branch
    {
        struct Endpoint
        {
            size_t sw;
            LeftRight side;
            size_t position;
            UpDown orientation;
        };
        std::vector<Endpoint> endpoints;
    };

    struct Switch
    {
        struct Germ
        {
            size_t branch;
            FirstSecond endpoint;
        };

        std::array<std::vector<Germ>, 2> connections;
    };

    struct ComplementaryRegion
    {
        unsigned int cusps, punctures;
    };

    struct Surface
    {
        unsigned int genus, punctures;
    };

    TrainTrack(size_t switches_count = 0, size_t branches_count = 0) : switches(switches_count), branches(branches_count)
    {
    }
    ~TrainTrack() = default;
    size_t add_switch();
    size_t add_branch();
    void attach_branch(size_t branch_index, size_t switch_index, LeftRight side);
    void attach_branch(size_t branch_index, size_t switch_index, LeftRight side, size_t pos);
    bool finalize(const TrainTrackOptions &options = TrainTrackOptions());
    Surface get_surface();

    template <std::integral T>
    std::vector<Measure<T>> get_vertex_measures() const;

    template <typename URBG>
    static TrainTrack random_trivalent_train_track(URBG &rng, size_t switches_count, const TrainTrackOptions &options = TrainTrackOptions());

    friend void to_json(nlohmann::json &j, const TrainTrack &tt);
    friend void from_json(const nlohmann::json &j, TrainTrack &tt);

private:
    bool is_finalized = false;
    std::vector<Switch> switches;
    std::vector<Branch> branches;
    std::vector<ComplementaryRegion> complementary_regions;
    Surface surface;

    void compute_complementary_regions(const TrainTrackOptions &options);
};

template <typename E>
constexpr E flip(E e);

NLOHMANN_JSON_SERIALIZE_ENUM(TrainTrack::LeftRight, {{TrainTrack::LeftRight::Left, 0}, {TrainTrack::LeftRight::Right, 1}});
NLOHMANN_JSON_SERIALIZE_ENUM(TrainTrack::UpDown, {{TrainTrack::UpDown::Up, "up"}, {TrainTrack::UpDown::Down, "down"}});
NLOHMANN_JSON_SERIALIZE_ENUM(TrainTrack::FirstSecond, {{TrainTrack::FirstSecond::First, 0}, {TrainTrack::FirstSecond::Second, 1}});
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(TrainTrack::Branch::Endpoint, sw, side, position, orientation);
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(TrainTrack::Branch, endpoints);
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(TrainTrack::Switch::Germ, branch, endpoint);
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(TrainTrack::Switch, connections);
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(TrainTrack::ComplementaryRegion, cusps, punctures);
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(TrainTrack::Surface, genus, punctures);

#include "train_track.cpp"