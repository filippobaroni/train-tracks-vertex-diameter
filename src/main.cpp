#include <thread>

#include "argparse/argparse.hpp"
#include "pcg_random.hpp"
#include "spdlog/spdlog.h"
#include "spdlog/fmt/ranges.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include "concurrentqueue/moodycamel/concurrentqueue.h"

#include "train_track.hpp"

#ifdef NDEBUG
constexpr unsigned int TRIES_FOR_SWITCH_NUMBER = 1'000'000;
constexpr unsigned int WORKER_UPDATE_FREQUENCY = 100'000;
constexpr unsigned int SWITCH_NUMBER_OBSOLETE_AFTER = 100'000;
#else
constexpr unsigned int TRIES_FOR_SWITCH_NUMBER = 100'000;
constexpr unsigned int WORKER_UPDATE_FREQUENCY = 10'000;
constexpr unsigned int SWITCH_NUMBER_OBSOLETE_AFTER = 10'000;
#endif
constexpr unsigned int AGGREGATOR_SLEEP_MILLISECONDS = 100;


int genus;
int punctures;
std::string output_directory;

std::vector<int> possible_switch_numbers;
std::mutex possible_switch_numbers_mutex;

std::atomic<unsigned int> best_intersection_number = 0;

// Determine numbers of switches compatible with surface topology.
// Since we only consider trivalent train tracks, the number of switches
// must be even. Also (see [Harer-Penner, Corollary 1.1.3]), the number of
// switches is at most 12g + 4p - 12. We generate TRIES_FOR_SWITCH_NUMBER
// random train tracks for each potential switch number, and we consider it valid
// if at least one of these train tracks lies on a surface of the correct
// topology.
void find_all_switch_numbers()
{
    pcg_extras::seed_seq_from<std::random_device> seed_source;
    pcg32 rng(seed_source);
    possible_switch_numbers.clear();
    constexpr int min_switches = 2;
    const int max_switches = 12 * genus + 4 * punctures - 12;
    for (int s = min_switches; s <= max_switches; s += 2)
    {
        for (int attempt = 0; attempt < TRIES_FOR_SWITCH_NUMBER; ++attempt)
        {
            const auto [g, p] = TrainTrack::random_trivalent_train_track(rng, s, {true}).get_surface();
            if (g == static_cast<unsigned int>(genus) && p == static_cast<unsigned int>(punctures))
            {
                possible_switch_numbers.push_back(s);
                spdlog::debug("{} switches is possible with probability {:.2f}%", s, 200.0 / (attempt + 2));
                break;
            }
            if (attempt + 1 == TRIES_FOR_SWITCH_NUMBER)
            {
                spdlog::debug("{} switches is impossible", s);
            }
        }
    }
}

// Structure for message from workers to the aggregator
struct MessageToAggregator
{
    std::map<int, unsigned int> attempts_by_switch_number;
    std::optional<std::tuple<unsigned int, TrainTrack, std::vector<CarriedCurvesConfiguration>>> new_result;
};

// Concurrent queue to hold messages from workers to the aggregator
moodycamel::ConcurrentQueue<MessageToAggregator> message_queue;


void worker(const std::stop_token& stop_token)
{
    // Random number generator
    pcg_extras::seed_seq_from<std::random_device> seed_source;
    pcg32 rng(seed_source);
    std::uniform_int_distribution<> switch_number_dist;

    // Own copy of possible switch numbers
    std::vector<int> own_possible_switch_numbers;

    // Statistics
    unsigned total_attempts_including_fails = 0;
    std::map<int, unsigned int> attempts_by_switch_number;

    // Function to update the aggregator and reset state
    const auto reset = [&]()
    {
        // Update own copy of possible switch numbers
        {
            std::scoped_lock lock(possible_switch_numbers_mutex);
            own_possible_switch_numbers = possible_switch_numbers;
        }
        switch_number_dist = std::uniform_int_distribution<>(0,
                                                             static_cast<int>(own_possible_switch_numbers.size()) -
                                                             1);
        // Reset attempt count
        total_attempts_including_fails = 0;
        attempts_by_switch_number.clear();
        for (const int s : own_possible_switch_numbers)
        {
            attempts_by_switch_number[s] = 0;
        }
    };
    // Run experiment
    reset();
    while (!stop_token.stop_requested() && !own_possible_switch_numbers.empty())
    {
        ++total_attempts_including_fails;
        // Select random switch number
        const int switch_number = own_possible_switch_numbers[switch_number_dist(rng)];
        // Generate random train track
        const auto train_track = TrainTrack::random_trivalent_train_track(rng, switch_number);
        // Does the underlying surface have the correct topology?
        if (const auto [thisGenus, thisPunctures] = train_track.get_surface(); thisGenus == static_cast<unsigned int>(
            genus) && thisPunctures == static_cast<unsigned int>(punctures))
        {
            // Increment attempts
            ++attempts_by_switch_number.at(switch_number);
            // Compute vertex curves
            const auto vertex_measures = train_track.get_vertex_measures<int>();
            // Compute pairwise intersection numbers
            unsigned int best_intersection_number_for_train_track = 0;
            std::vector<CarriedCurvesConfiguration> best_configurations;
            for (unsigned int i = 0; i < vertex_measures.size(); ++i)
            {
                for (unsigned int j = i + 1; j < vertex_measures.size(); ++j)
                {
                    const auto [intersections, configuration] =
                        train_track.compute_intersection_number(vertex_measures[i], vertex_measures[j]);
                    if (intersections > best_intersection_number_for_train_track)
                    {
                        best_intersection_number_for_train_track = intersections;
                        best_configurations.clear();
                    }
                    if (intersections == best_intersection_number_for_train_track)
                    {
                        best_configurations.emplace_back(std::move(configuration));
                    }
                }
            }
            // If this matches or improves the current best, send results to the aggregator
            if (best_intersection_number_for_train_track >= best_intersection_number)
            {
                message_queue.enqueue(MessageToAggregator{
                    attempts_by_switch_number, std::tuple{
                        best_intersection_number_for_train_track,
                        train_track, best_configurations
                    }
                });
                reset();
                continue;
            }
        }
        if (total_attempts_including_fails % WORKER_UPDATE_FREQUENCY == 0)
        {
            // Periodically reset
            message_queue.enqueue(MessageToAggregator{
                attempts_by_switch_number, std::nullopt
            });
            reset();
        }
    }
}

struct Checkpoint
{
    int genus;
    int punctures;
    unsigned int best_intersection_number;
    std::vector<int> possible_switch_numbers;
    std::map<int, unsigned int> attempts_by_switch_number;
};

void aggregator(std::map<int, unsigned int> attempts_by_switch_number,
                std::map<int, unsigned int> attempts_without_state_of_the_art)
{
    MessageToAggregator message;
    while (true)
    {
        while (message_queue.try_dequeue(message))
        {
            bool has_state_of_the_art = false;
            if (message.new_result.has_value())
            {
                const auto& [intersections, train_track, configurations] = message.new_result.value();
                spdlog::debug(
                    "Received message; new_attempts: {}, new_result: {} configurations with intersection number {}",
                    message.attempts_by_switch_number, configurations.size(), intersections);
                if (intersections > best_intersection_number)
                {
                    best_intersection_number = intersections;
                    spdlog::info("New best intersection number found: {}", best_intersection_number.load());
                }
                if (intersections == best_intersection_number)
                {
                    has_state_of_the_art = true;
                }
            }
            else
            {
                spdlog::debug("Received message; new_attempts: {}, no new result",
                              message.attempts_by_switch_number);
            }
            // Update attempts
            for (const auto& [s, count] : message.attempts_by_switch_number)
            {
                if (std::ranges::contains(possible_switch_numbers, s))
                {
                    attempts_by_switch_number[s] += count;
                    if (!has_state_of_the_art)
                    {
                        if ((attempts_without_state_of_the_art[s] += count) >= SWITCH_NUMBER_OBSOLETE_AFTER)
                        {
                            {
                                std::scoped_lock lock(possible_switch_numbers_mutex);
                                std::erase(possible_switch_numbers, s);
                            }
                            spdlog::info("Will no longer check train tracks with {} switches", s);
                        }
                    }
                }
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(AGGREGATOR_SLEEP_MILLISECONDS));
    }
}


int main(const int argc, const char** argv)
{
    int thread_count;

    argparse::ArgumentParser run_command("run");
    run_command.add_description("Start new search");
    run_command.add_argument("-g", "--genus")
               .required()
               .scan<'i', int>()
               .metavar("GENUS")
               .help("Genus of the underlying surface (>= 0)")
               .store_into(genus);
    run_command.add_argument("-p", "--punctures")
               .scan<'i', int>()
               .default_value(0)
               .metavar("PUNCTURES")
               .help("Number of punctures of the underlying surface (>= 0)")
               .store_into(punctures);
    run_command.add_argument("-t", "--threads")
               .scan<'i', int>()
               .default_value(std::max(1, static_cast<int>(std::thread::hardware_concurrency()) - 2))
               .metavar("N")
               .help("Number of threads to use (>= 1)")
               .store_into(thread_count);
    run_command.add_argument("-o", "--output")
               .default_value(std::string("output"))
               .metavar("DIR")
               .help("Output directory (must not exist)")
               .store_into(output_directory);


    argparse::ArgumentParser program("Train track vertex explorer");
    program.add_subparser(run_command);

    try
    {
        program.parse_args(argc, argv);
    }
    catch (const std::exception& err)
    {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        std::exit(1);
    }

    // Create console logger
    const auto console_logger = spdlog::stdout_color_mt("console");
    console_logger->set_level(spdlog::level::info);
    spdlog::set_default_logger(console_logger);

    // Depending on used command
    if (program.is_subcommand_used(run_command))
    {
        // Set up new experiment
        spdlog::info("Starting new search (genus={}, punctures={})", genus, punctures);
        spdlog::info("Using {} threads", thread_count);
        spdlog::info("Results will be saved to '{}'", output_directory);

        spdlog::info("Determining possible switch numbers...");
        find_all_switch_numbers();
        spdlog::info("Found {}", possible_switch_numbers);
    }
    std::vector<std::jthread> workers;
    workers.reserve(thread_count);
    for (int i = 0; i < thread_count; ++i)
    {
        workers.emplace_back(worker);
    }

    aggregator({}, {});


    return 0;
}
