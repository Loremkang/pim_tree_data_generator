#include <sys/mman.h>
#include <sys/stat.h>

#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>

#include "fcntl.h"

using namespace std;

#include <parlay/primitives.h>
#include <parlay/sequence.h>

#include <argparse/argparse.hpp>

#include "host/operation_def.hpp"
#include "host/oracle.hpp"
#include "zipfian_generator.hpp"

const size_t scan_factor = 100;
const size_t random_resolution = 1048576;
const double EPS = 1e-12;
const size_t kNumPartitions = 64;
const size_t kPrintTopK = 10;

template <typename T, bool check_sorted_deduplicated = false>
std::vector<T> LoadKeysFromBinary(const std::string& file_path,
                                  const uint64_t start, const uint64_t end) {
    std::ifstream file(file_path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open file: " + file_path);
    } else {
        std::cout << "Open File: " << file_path << std::endl;
    }

    uint64_t element_count;
    uint64_t element_size;

    file.read(reinterpret_cast<char*>(&element_count), sizeof(uint64_t));
    std::cout << "File has " << element_count << " elements" << std::endl;

    assert(end > start);

    uint64_t read_count = std::min(element_count, end - start);
    std::cout << "Read " << read_count << " elements from " << start << " to "
              << end << std::endl;

    file.read(reinterpret_cast<char*>(&element_size), sizeof(uint64_t));
    std::cout << "Each element has " << element_size << " bytes" << std::endl;
    assert(element_size == sizeof(T));

    uint64_t read_size = read_count * element_size;
    std::cout << "Read " << read_size << " bytes" << std::endl;

    std::vector<T> data(read_count);

    file.seekg(sizeof(uint64_t) * 2 + start * element_size, std::ios::beg);
    file.read(reinterpret_cast<char*>(data.data()), read_size);

    if (check_sorted_deduplicated) {
        parlay::parallel_for(1, read_count,
                             [&](size_t i) { assert(data[i] > data[i - 1]); });
    }

    return data;
}

template <typename T>
void StoreElementsToBinary(const std::string& file_path,
                           const std::vector<T>& data) {
    if (std::ifstream(file_path)) {
        throw std::runtime_error("File already exists: " + file_path);
    }

    std::ofstream file(file_path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open file: " + file_path);
    }

    uint64_t element_count = data.size();
    uint64_t element_size = sizeof(T);
    uint64_t total_size = element_count * element_size;

    file.write(reinterpret_cast<const char*>(&element_count), sizeof(uint64_t));
    file.write(reinterpret_cast<const char*>(&element_size), sizeof(uint64_t));
    file.write(reinterpret_cast<const char*>(data.data()), total_size);
}

std::vector<uint64_t> UniformRandomKeys(size_t size, size_t seed, bool sort) {
    std::vector<uint64_t> ret(size);
    parlay::parallel_for(0, size,
                         [&](size_t i) { ret[i] = parlay::hash64(seed + i); });
    ret[0] = 0;
    if (sort) {
        parlay::sort_inplace(ret);
    }
    return ret;
}

std::vector<key_value> UniformRandomKVs(size_t size, size_t seed, bool sort) {
    std::vector<key_value> ret(size);
    parlay::parallel_for(0, size, [&](size_t i) {
        ret[i].key = parlay::hash64(seed + i);
        ret[i].value = parlay::hash64(ret[i].key);
    });
    ret[0] = {0, parlay::hash64(0)};
    if (sort) {
        parlay::sort_inplace(ret);
    }
    return ret;
}

std::vector<key_value> LoadOrInitKVs(std::string file_name, size_t length) {
    assert(file_name != "");

    struct stat buffer;
    std::vector<key_value> kvs;
    if (stat(file_name.c_str(), &buffer) == 0) {
        kvs = LoadKeysFromBinary<key_value, true>(file_name, 0, length);
        std::cout << "Loaded " << kvs.size() << " keys from " << file_name
              << std::endl;
    } else {
        // Initialize keys with some logic if file does not exist
        kvs = UniformRandomKVs(length, 0, true);
        std::cout << "Generated " << kvs.size() << " keys" << std::endl;
        StoreElementsToBinary<key_value>(file_name, kvs);
        std::cout << "Generated keys stored to " << file_name << std::endl;
    }

    // Ensure kvs has a key with value 0
    assert(std::any_of(kvs.begin(), kvs.end(),
                       [](const key_value& kv) { return kv.key == 0; }) &&
           "kvs must contain a key with value 0");

    return kvs;
}

size_t GetBatchType(vector<double> possibility, size_t idx, size_t seed = 0) {
    std::vector<double> prefix_sum(possibility.size(), 0.0);
    prefix_sum[0] = possibility[0];
    for (size_t i = 1; i < possibility.size(); ++i) {
        prefix_sum[i] = prefix_sum[i - 1] + possibility[i];
    }
    assert(abs(prefix_sum[prefix_sum.size() - 1] - 1.0) < EPS &&
           "Possibility should sum to 1");

    size_t rnd = parlay::hash64(idx + seed) % random_resolution;
    double r = (double)rnd / random_resolution;

    size_t batch_type =
        std::lower_bound(prefix_sum.begin(), prefix_sum.end(), r) -
        prefix_sum.begin();
    assert(batch_type > 0 && batch_type < kNumTypesOfOperations &&
           "Invalid batch type");
    return batch_type;
}

void GenerateGetBatch(union_operation* target, size_t batch_size,
                      operation_t batch_type, double alpha,
                      ZipfianGenerator& zipf_gen, size_t kNumPartitions,
                      map_oracle& oracle, size_t seed = 0) {
    auto distribution_in_partitions = zipf_gen.Generate(batch_size, seed);
    parlay::parallel_for(0, batch_size, [&](size_t i) {
        auto [start, size] =
            zipf_gen.KeyRangeForPartition(distribution_in_partitions[i]);
        uint64_t rand_gen_key = start + parlay::hash64(i + seed) % size;
        uint64_t get_keys = oracle.predecessor(rand_gen_key).key;

        target[i].type = batch_type;
        target[i].tsk.g.key = get_keys;
        assert(oracle.predecessor(get_keys).key == get_keys);
    });
    zipf_gen.PrintTopKHotestPartition(batch_size, [&](size_t i) {
        return target[i].tsk.g.key;
    }, kPrintTopK);
}

void GeneratePredecessorBatch(union_operation* target, size_t batch_size,
                              operation_t batch_type, double alpha,
                              ZipfianGenerator& zipf_gen, size_t kNumPartitions,
                              map_oracle& oracle, size_t seed = 0) {
    auto distribution_in_partitions = zipf_gen.Generate(batch_size, seed);
    parlay::parallel_for(0, batch_size, [&](size_t i) {
        auto [start, size] =
            zipf_gen.KeyRangeForPartition(distribution_in_partitions[i]);
        uint64_t rand_gen_key = start + parlay::hash64(i + seed) % size;
        target[i].type = batch_type;
        target[i].tsk.p.key = rand_gen_key;
    });
    zipf_gen.PrintTopKHotestPartition(batch_size, [&](size_t i) {
        return target[i].tsk.p.key;
    }, kPrintTopK);
}

void GenerateInsertBatch(union_operation* target, size_t batch_size,
                         operation_t batch_type, double alpha,
                         ZipfianGenerator& zipf_gen, size_t kNumPartitions,
                         map_oracle& oracle, size_t seed = 0) {
    auto distribution_in_partitions = zipf_gen.Generate(batch_size, seed);
    parlay::parallel_for(0, batch_size, [&](size_t i) {
        auto [start, size] =
            zipf_gen.KeyRangeForPartition(distribution_in_partitions[i]);
        uint64_t rand_gen_key = start + parlay::hash64(i + seed) % size;
        target[i].type = batch_type;
        target[i].tsk.i.key = rand_gen_key;
        target[i].tsk.i.value = parlay::hash64(rand_gen_key);
    });
    oracle.insert_batch(batch_size, [&](size_t i) {
        key_value kv;
        kv.key = target[i].tsk.i.key;
        kv.value = target[i].tsk.i.value;
        return kv;
    });
    zipf_gen.PrintTopKHotestPartition(batch_size, [&](size_t i) {
        return target[i].tsk.i.key;
    }, kPrintTopK);
}

void GenerateRemoveBatch(union_operation* target, size_t batch_size,
                         operation_t batch_type, double alpha,
                         ZipfianGenerator& zipf_gen, size_t kNumPartitions,
                         map_oracle& oracle, size_t seed = 0) {
    auto distribution_in_partitions = zipf_gen.Generate(batch_size, seed);
    parlay::parallel_for(0, batch_size, [&](size_t i) {
        auto [start, size] =
            zipf_gen.KeyRangeForPartition(distribution_in_partitions[i]);
        uint64_t rand_gen_key = start + parlay::hash64(i + seed) % size;
        uint64_t remove_key = oracle.predecessor(rand_gen_key).key;
        assert(remove_key != 0);
        target[i].type = batch_type;
        target[i].tsk.r.key = remove_key;
        assert(oracle.predecessor(remove_key).key == remove_key);
    });
    zipf_gen.PrintTopKHotestPartition(batch_size, [&](size_t i) {
        return target[i].tsk.r.key;
    }, kPrintTopK);
    oracle.remove_batch(batch_size,
                        [&](size_t i) { return target[i].tsk.r.key; });
}

void GenerateScanBatch(union_operation* target, size_t batch_size,
                       operation_t batch_type, double alpha,
                       ZipfianGenerator& zipf_gen, size_t kNumPartitions,
                       map_oracle& oracle, size_t seed = 0) {
    size_t scan_length = UINT64_MAX / oracle.size() * scan_factor;

    auto distribution_in_partitions = zipf_gen.Generate(batch_size, seed);
    parlay::parallel_for(0, batch_size, [&](size_t i) {
        auto [start, size] =
            zipf_gen.KeyRangeForPartition(distribution_in_partitions[i]);
        uint64_t rand_gen_key = start + parlay::hash64(i + seed) % size;
        uint64_t scan_key = rand_gen_key;
        target[i].type = batch_type;
        target[i].tsk.s.lkey = scan_key;
        target[i].tsk.s.rkey = scan_key + scan_length;
        assert(UINT64_MAX - scan_key >= scan_length);
    });
    zipf_gen.PrintTopKHotestPartition(batch_size, [&](size_t i) {
        return target[i].tsk.s.lkey;
    }, kPrintTopK);
}

std::vector<union_operation> GenerateInitFile(std::string file_name,
                                              size_t batch_size,
                                              std::vector<key_value>& kvs) {
    assert(kvs.size() % batch_size == 0 &&
           "ops_size should be multiple of batch_size");

    std::cout << "KVs size: " << kvs.size() << std::endl;

    // shuffle
    for (size_t i = 0; i < kvs.size(); i ++) {
        size_t swap_pos = parlay::hash64(i) % (i + 1);
        std::swap(kvs[i], kvs[swap_pos]);
    }

    std::cout << "shuffle finished" << std::endl;

    std::vector<union_operation> operations(kvs.size());
    parlay::parallel_for(0, kvs.size(), [&](size_t i) {
        operations[i].type = operation_t::insert_t;
        operations[i].tsk.i.key = kvs[i].key;
        operations[i].tsk.i.value = kvs[i].value;
    });

    return operations;
}

std::vector<union_operation> GenerateTestFile(
    std::string file_name, bool init_file, size_t batch_size, size_t ops_size,
    double alpha, size_t kNumPartitions, std::vector<key_value>& kvs,
    std::vector<double>& possibilities) {
    assert(ops_size % batch_size == 0 &&
           "ops_size should be multiple of batch_size");

    map_oracle oracle;
    oracle.init(kvs.size(), [&](size_t i) { return kvs[i]; });

    std::cout << "Oracle size: " << oracle.size() << std::endl;

    size_t batch_size_scan = batch_size / scan_factor;
    auto zipf_gen = ZipfianGenerator(kNumPartitions, alpha);
    std::vector<union_operation> operations(ops_size);
    for (size_t i = 0; i < ops_size; i += batch_size) {
        operation_t batch_type =
            static_cast<operation_t>(GetBatchType(possibilities, i));
        std::cout << "Generating batch " << i << " with type " << batch_type
                  << std::endl;
        switch (batch_type) {
            case operation_t::get_t:
                GenerateGetBatch(operations.data() + i, batch_size, batch_type,
                                 alpha, zipf_gen, kNumPartitions,
                                 oracle, i);
                break;
            case operation_t::predecessor_t:
                GeneratePredecessorBatch(operations.data() + i, batch_size,
                                         batch_type, alpha, zipf_gen,
                                         kNumPartitions, oracle, i);
                break;
            case operation_t::insert_t:
                GenerateInsertBatch(operations.data() + i, batch_size,
                                    batch_type, alpha, zipf_gen,
                                    kNumPartitions, oracle, i);
                break;
            case operation_t::remove_t:
                GenerateRemoveBatch(operations.data() + i, batch_size,
                                    batch_type, alpha, zipf_gen,
                                    kNumPartitions, oracle, i);
                break;
            case operation_t::scan_t:
                GenerateScanBatch(operations.data() + i, batch_size_scan,
                                  batch_type, alpha, zipf_gen,
                                  kNumPartitions, oracle, i);
                break;
            default:
                assert(false && "Invalid batch type");
        }
        std::cout << "Generated batch " << i << std::endl;
    }

    return operations;
}

void InitParser(argparse::ArgumentParser& cli_parser) {
    cli_parser.add_argument("--from", "-f")
        .required()
        .help("--from [kv_file_i64]");
    cli_parser.add_argument("--kv_length", "-k")
        .required()
        .help("--kv_length [kv_file_length]")
        .scan<'i', int>();
    cli_parser.add_argument("--length", "-l")
        .required()
        .help("--length [length_of_the_target_operation_sequence]")
        .scan<'i', int>();
    cli_parser.add_argument("--batch_size", "-b")
        .required()
        .help("--batch_size [batch size]")
        .scan<'i', int>();
    cli_parser.add_argument("--alpha", "-a")
        .required()
        .help("--alpha [?x]")
        .scan<'g', double>();
    cli_parser.add_argument("--output", "-o")
        .required()
        .help("--output [test_file]");
    cli_parser.add_argument("--init_file")
        .help("generate all insert operation sequence of input file")
        .default_value(false)
        .implicit_value(true);

    cli_parser.add_argument("--get", "-g")
        .help("-g [get_ratio]")
        .default_value(0.0)
        .scan<'g', double>();
    cli_parser.add_argument("--update", "-u")
        .help("-u [update_ratio]")
        .default_value(0.0)
        .scan<'g', double>();
    cli_parser.add_argument("--predecessor", "-p")
        .help("-p [predecessor_ratio]")
        .default_value(0.0)
        .scan<'g', double>();
    cli_parser.add_argument("--insert", "-i")
        .help("-i [insert_ratio]")
        .default_value(0.0)
        .scan<'g', double>();
    cli_parser.add_argument("--remove", "-r")
        .help("-r [remove_ratio]")
        .default_value(0.0)
        .scan<'g', double>();
    cli_parser.add_argument("--scan", "-s")
        .help("-s [scan_ratio]")
        .default_value(0.0)
        .scan<'g', double>();
}

int main(int argc, char* argv[]) {
    // ./data_generator
    argparse::ArgumentParser cli_parser;
    InitParser(cli_parser);

    try {
        cli_parser.parse_args(argc, argv);
    } catch (const std::runtime_error& err) {
        std::cerr << err.what() << std::endl;
        std::cerr << cli_parser;
        std::exit(1);
    }

    std::string kv_file_name = cli_parser.get<std::string>("-f");
    size_t kv_length = cli_parser.get<int>("-k");
    std::vector<key_value> kvs = LoadOrInitKVs(kv_file_name, kv_length);

    size_t test_length = cli_parser.get<int>("-l");
    size_t batch_size = cli_parser.get<int>("-b");

    double zipfian_alpha = cli_parser.get<double>("-a");
    std::string output_file_name = cli_parser.get<std::string>("-o");
    bool init_file = cli_parser.get<bool>("--init_file");

    assert(zipfian_alpha > -EPS);

    vector<double> posibility_of_tasks(kNumTypesOfOperations);
    posibility_of_tasks[1] = cli_parser.get<double>("-g");
    posibility_of_tasks[2] = cli_parser.get<double>("-u");
    posibility_of_tasks[3] = cli_parser.get<double>("-p");
    posibility_of_tasks[4] = cli_parser.get<double>("-i");
    posibility_of_tasks[5] = cli_parser.get<double>("-r");
    posibility_of_tasks[6] = cli_parser.get<double>("-s");

    assert(posibility_of_tasks[2] < EPS && "update operation is not supported");

    std::vector<union_operation> operations;
    if (init_file) {
        assert(test_length == kv_length);
        assert(zipfian_alpha < EPS);
        operations = GenerateInitFile(output_file_name, batch_size, kvs);
    } else {
        operations = GenerateTestFile(output_file_name, false, batch_size,
                                      test_length, zipfian_alpha,
                                      kNumPartitions, kvs, posibility_of_tasks);
    }

    std::cout << "Generated " << operations.size() << " operations" << std::endl;

    std::string kv_file_name_hint = kv_file_name;
    std::replace(kv_file_name_hint.begin(), kv_file_name_hint.end(), '/', '-');

    stringstream ss;
    std::string actual_file_name;
    std::string is_init_file = init_file ? "_[INIT]" : "_[----]";
    ss << output_file_name << "_key=[" << kv_file_name_hint
       << "]_keylength=" << kv_length << "_P=" << kNumPartitions << is_init_file
       << "_batch=" << batch_size << "_alpha=" << std::fixed
       << std::setprecision(1) << zipfian_alpha << "_size=" << test_length;
    for (double p : posibility_of_tasks) {
        ss << "_" << std::fixed << std::setprecision(1) << p;
    }
    actual_file_name = ss.str();
    ss >> actual_file_name;
    std::cout << "Write to " << actual_file_name << std::endl;

    StoreElementsToBinary<union_operation>(actual_file_name, operations);
}