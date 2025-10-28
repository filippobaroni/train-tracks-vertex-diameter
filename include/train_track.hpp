#pragma once
#include <vector>
#include <utility>
#include <concepts>
#include <array>
#include <optional>
#include <random>
#include <nlohmann/json.hpp>

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

template <std::integral T>
using Measure = std::vector<T>;

using CarriedCurvesConfiguration = std::vector<std::vector<FirstSecond>>;

template <std::integral T>
CarriedCurvesConfiguration configuration_from_carried_curves(const Measure<T> &, const Measure<T> &);

template <std::integral T, typename RNG>
CarriedCurvesConfiguration random_configuration_from_carried_curves(RNG &, const Measure<T> &, const Measure<T> &);

struct TrainTrackOptions
{
    bool add_punctures = true;
};

class TrainTrack
{
public:
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
        struct BranchSide
        {
            size_t branch;
            UpDown side;
        };
        unsigned int cusps, punctures;
        std::vector<BranchSide> boundary;
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
    const auto &get_switches() const;
    const auto &get_branches() const;
    const auto &get_surface() const;

    template <std::integral T>
    std::vector<Measure<T>> get_vertex_measures() const;

    template <std::integral T>
    std::tuple<unsigned long, CarriedCurvesConfiguration> compute_intersection_number(const Measure<T> &m1, const Measure<T> &m2) const;
    unsigned long compute_intersection_number(CarriedCurvesConfiguration &) const;

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

class ConfigurationSwitchEndpointsRange
{
public:
    ConfigurationSwitchEndpointsRange(const TrainTrack &, const CarriedCurvesConfiguration &, size_t, LeftRight);
    struct Iterator
    {
        struct value_type
        {
            size_t connection_position;
            int position_on_branch;
        };
        using reference = value_type const &;
        using pointer = value_type const *;
        using difference_type = std::ptrdiff_t;
        using iterator_category = std::input_iterator_tag;

        Iterator(const TrainTrack &, const CarriedCurvesConfiguration &, size_t, LeftRight, size_t, int);
        constexpr Iterator(Iterator const &) = default;
        reference operator*() const;
        pointer operator->() const;
        Iterator &operator++();
        Iterator operator++(int);
        Iterator &operator=(const Iterator &);
        bool operator==(const Iterator &) const;
        bool operator!=(const Iterator &) const;

        Iterator &skip_empty_branches();

        const TrainTrack &train_track;
        const CarriedCurvesConfiguration &configuration;
        size_t switch_index;
        LeftRight side;
        value_type value;
    };

    Iterator begin() const;
    Iterator end() const;

private:
    const TrainTrack &train_track;
    const CarriedCurvesConfiguration &configuration;
    size_t switch_index;
    LeftRight side;
};

unsigned long intersections_in_configuration(const TrainTrack &, const CarriedCurvesConfiguration &);

#include "train_track.cpp"