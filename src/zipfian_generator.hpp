#pragma once

#include <parlay/primitives.h>
#include <parlay/utilities.h>
#include <parlay/sequence.h>
#include <vector>
#include <cmath>

// generate zipfian distributed element indexes in [0, n) according to the given alpha
class ZipfianGenerator {
    double alpha_;
    int num_partition_;
    std::vector<double> zipf_possibility_, zipf_possibility_aggregated_;

    uint64_t range_size;
    const size_t random_resolution = 1048576;

public:
    ZipfianGenerator(int n, double alpha) : num_partition_(n), alpha_(alpha) {
        double sum = 0;
        zipf_possibility_.resize(n, 1.0);
        zipf_possibility_aggregated_.resize(n, 0.0);
        for (int i = 0; i < n; i++) {
            zipf_possibility_[i] /= pow(i + 1, alpha);
            sum += zipf_possibility_[i];
        }
        for (int i = 0; i < n; i++) {
            zipf_possibility_[i] /= sum;
        }
        zipf_possibility_aggregated_[0] = zipf_possibility_[0];

        for (int i = 1; i < n; i++) {
            zipf_possibility_aggregated_[i] = zipf_possibility_aggregated_[i - 1] + zipf_possibility_[i];
        }

        range_size = UINT64_MAX / num_partition_ + 1;
        assert((num_partition_ - 1) <= (UINT64_MAX / range_size));
        assert(UINT64_MAX - (range_size * (num_partition_ - 1)) + 1 <= range_size);

        // Print the possibilities
        for (int i = 0; i < 5; i++) {
            std::cout << "Index " << i << " possibility: " << zipf_possibility_[i] << " reversed: " << 1.0 / zipf_possibility_[i] << std::endl;
        }
        // Print range size
        std::cout << "Range size: " << range_size << std::endl;
    }

    // return (l, size) for range[l, l + size)
    std::pair<uint64_t, uint64_t> KeyRangeForPartition(int partition_id) {
        uint64_t start = range_size * partition_id;
        if (UINT64_MAX - start < range_size) {
            return {start, UINT64_MAX - start + 1};
        } else {
            return {start, range_size};
        }
    }

    size_t FindPartition(uint64_t key) {
        return key / range_size;
    }

    std::vector<size_t> Generate(int m, size_t seed = 0, bool shuffle = true) {
        std::vector<size_t> res(m);
        std::vector<size_t> shuffled_index(num_partition_);
        for (size_t i = 0; i < num_partition_; i ++) {
            shuffled_index[i] = i;
            if (shuffle) {
                size_t rnd = parlay::hash64(i + seed) % (i + 1);
                std::swap(shuffled_index[i], shuffled_index[rnd]);
            }
        }

        parlay::parallel_for(0, m, [&](int i) {
            size_t rnd = parlay::hash64(i + seed) % random_resolution;
            double r = (double)rnd / random_resolution;
            int idx = std::lower_bound(zipf_possibility_aggregated_.begin(),
                                       zipf_possibility_aggregated_.end(), r) -
                      zipf_possibility_aggregated_.begin();
            res[i] = shuffled_index[idx];
        });
        return res;
    }

    template<typename GetKeyF>
    void PrintTopKHotestPartition(size_t size, GetKeyF get_key, size_t k) {
        assert(k <= num_partition_);
        static_assert(std::is_same<uint64_t, decltype(get_key(0))>::value);

        std::vector<size_t> partition_count(num_partition_, 0);
        for (size_t i = 0; i < size; i ++) {
            partition_count[FindPartition(get_key(i))]++;
        }
        
        std::vector<std::pair<size_t, size_t>> partition_count_with_id(num_partition_);
        for (size_t i = 0; i < num_partition_; i ++) {
            partition_count_with_id[i] = {partition_count[i], i};
        }
        // for (const auto& p : partition_count_with_id) {
        //     std::cout << "Partition " << p.second << " count: " << p.first << std::endl;
        // }

        std::sort(partition_count_with_id.begin(), partition_count_with_id.end(), std::greater<>());
        
        for (size_t i = 0; i < k; i ++) {
            std::cout << "Partition " << partition_count_with_id[i].second
                      << " has " << std::fixed << std::setprecision(2)
                      << (double)partition_count_with_id[i].first / size * 100 << "% keys" << std::endl;
        }
    }
};
