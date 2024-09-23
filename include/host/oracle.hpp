#pragma once

#include <parlay/primitives.h>
#include <parlay/range.h>
#include <parlay/sequence.h>
#include <cstring>
#include <vector>
#include <map>
#include "operation_def.hpp"
#include "value.hpp"
using namespace std;

// class batch_parallel_oracle_base {
//     public:
//     virtual parlay::sequence<key_value> predecessor_batch(uint64_t* keys, size_t size) = 0;
//     virtual void insert_batch(insert_operation* insert_ops, size_t size) = 0;
//     virtual void remove_batch(remove_operation* remove_ops, size_t size) = 0;
//     virtual parlay::sequence<std::pair<int64_t, int64_t>> scan_size_batch(scan_operation* scan_ops, size_t size) = 0;
//     virtual parlay::sequence<key_value> dump() = 0;
//     virtual size_t size() = 0;
// };

class map_oracle {
    public:
    std::map<uint64_t, uint64_t> data;
    std::vector<uint64_t> keys;

    template<typename GetKVF>
    void init(size_t size, GetKVF get_kv) {
        assert(get_kv(0).key == 0);
        size_t load_batch_size = 1e6;
        for (size_t i = 0; i < size; i += load_batch_size) {
            size_t load_size = std::min(load_batch_size, size - i);
            insert_batch(load_size, [&](size_t j) { return get_kv(i + j); });
            std::cout << "Loaded " << i + load_size << " keys" << std::endl;
        }
    }

    key_value predecessor(uint64_t key) {
        auto itr = data.upper_bound(key);
        assert(itr != data.begin());
        itr--;
        return (key_value){.key = itr->first, .value = itr->second};
    }
    
    template<typename GetKeyF>
    std::vector<key_value> predecessor_batch(size_t size, GetKeyF get_key) {
        static_assert(
            std::is_same<typename std::uint64_t, decltype(get_key(0))>::value);

        std::vector<key_value> ret(size);
        parlay::parallel_for(0, size, [&](size_t i) {
            auto itr = data.upper_bound(get_key(i));
            itr--;
            ret[i] = (key_value){.key = itr->first, .value = itr->second};
        });
        return ret;
    }

    template<typename GetKVF>
    void insert_batch(size_t size, GetKVF get_kv) {
        for (size_t i = 0; i < size; i++) {
            key_value kv = get_kv(i);
            data[kv.key] = kv.value;
        }
    }

    template<typename GetKeyF>
    void remove_batch(size_t size, GetKeyF get_key) {
        for (size_t i = 0; i < size; i++) {
            uint64_t key = get_key(i);
            auto itr = data.find(key);
            if (itr != data.end()) {
                data.erase(itr);
            }
        }
    }

    template<typename GetLRKeyF>
    std::vector<std::vector<key_value>> scan_size_batch(size_t size, GetLRKeyF get_lrkey) {
        std::vector<std::vector<key_value>> ret(size);
        parlay::parallel_for(0, size, [&](size_t i) {
            uint64_t lkey, rkey;
            std::tie(lkey, rkey) = get_lrkey(i);
            auto lpos = data.lower_bound(lkey);
            auto rpos = data.upper_bound(rkey);
            for (auto itr = lpos; itr != rpos; itr++) {
                ret[i].push_back((key_value){.key = itr->first, .value = itr->second});
            }
        });
        return ret;
    }

    size_t size() {
        return data.size();
    }

    std::vector<key_value> dump() {
        size_t cnt = 0;
        size_t size = data.size();
        std::vector<key_value> ret(size);
        for (auto itr = data.begin(); itr != data.end(); itr++) {
            ret[cnt++] = (key_value){.key = itr->first, .value = itr->second};
        }
        return ret;
    }
};