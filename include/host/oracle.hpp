#pragma once

#include <parlay/primitives.h>
#include <parlay/range.h>
#include <parlay/sequence.h>
#include <cstring>
#include <vector>
#include <map>
#include "operation_def.hpp"
#include "value.hpp"

// class batch_parallel_oracle_base {
//     public:
//     virtual parlay::sequence<key_value> predecessor_batch(uint64_t* keys, size_t size) = 0;
//     virtual void insert_batch(insert_operation* insert_ops, size_t size) = 0;
//     virtual void remove_batch(remove_operation* remove_ops, size_t size) = 0;
//     virtual parlay::sequence<std::pair<int64_t, int64_t>> scan_size_batch(scan_operation* scan_ops, size_t size) = 0;
//     virtual parlay::sequence<key_value> dump() = 0;
//     virtual size_t size() = 0;
// };

class Oracle {
    public:
    std::map<int64_t, int64_t> data;
    std::vector<int64_t> keys;

    void Init() {
        data.clear();
        keys.clear();
        data[INT64_MIN] = INT64_MIN;
    }

    key_value Predecessor(int64_t key) {
        auto itr = data.upper_bound(key);
        assert(itr != data.begin());
        itr--;
        return (key_value)(*itr);
    }
    
    template <typename GetKeyF>
    std::vector<key_value> RunBatchGet(size_t size, GetKeyF get_key) {
        static_assert(
            std::is_same<int64_t, decltype(get_key(0))>::value, "get_key(0) must return int64_t");
        std::vector<key_value> ret(size);
        parlay::parallel_for(0, size, [&](size_t i) {
            int64_t key = get_key(i);
            auto itr = data.find(key);
            if (itr != data.end()) {
                ret[i] = (key_value)(*itr);
            } else {
                ret[i] = (key_value)(INT64_MIN, INT64_MIN);
            }
        });
        return ret;
    }

    template<typename GetKeyF>
    std::vector<key_value> RunBatchPredecessor(size_t size, GetKeyF get_key) {
        static_assert(
            std::is_same<int64_t, decltype(get_key(0))>::value, "get_key(0) must return int64_t");

        std::vector<key_value> ret(size);
        parlay::parallel_for(0, size, [&](size_t i) {
            auto itr = data.upper_bound(get_key(i));
            itr--;
            ret[i] = (key_value)(*itr);
        });
        return ret;
    }

    template<typename GetKVF>
    void RunBatchInsert(size_t size, GetKVF get_kv) {
        static_assert(
            std::is_same<key_value, decltype(get_kv(0))>::value, "get_kv(0) must return key_value");
        std::vector<key_value> kvs(size);
        parlay::parallel_for(0, size, [&](size_t i) {
            kvs[i] = get_kv(i);
        });
        parlay::sort_inplace(kvs);
        for (size_t i = 0; i < size; i++) {
            key_value kv = kvs[i];
            if (i > 0 && kvs[i - 1].key == kv.key) {
                continue;
            } else {
                data[kv.key] = kv.value;
            }
        }
    }

    template<typename GetKeyF>
    void RunBatchRemove(size_t size, GetKeyF get_key) {
        static_assert(
            std::is_same<int64_t, decltype(get_key(0))>::value, "get_key(0) must return int64_t");
        for (size_t i = 0; i < size; i++) {
            int64_t key = get_key(i);
            auto itr = data.find(key);
            if (itr != data.end()) {
                data.erase(itr);
            }
        }
    }

    template<typename GetLRKeyF>
    std::vector<std::vector<key_value>> RunBatchScan(size_t size, GetLRKeyF get_lrkey) {
        std::vector<std::vector<key_value>> ret(size);
        parlay::parallel_for(0, size, [&](size_t i) {
            int64_t lkey, rkey;
            std::tie(lkey, rkey) = get_lrkey(i);
            auto lpos = data.lower_bound(lkey);
            auto rpos = data.upper_bound(rkey);
            for (auto itr = lpos; itr != rpos; itr++) {
                ret[i].push_back((key_value)(*itr));
            }
        });
        return ret;
    }

    size_t Size() {
        return data.size();
    }

    std::vector<key_value> Dump() {
        size_t cnt = 0;
        size_t size = data.size();
        std::vector<key_value> ret(size);
        for (auto itr = data.begin(); itr != data.end(); itr++) {
            ret[cnt++] = (key_value)(*itr);
        }
        return ret;
    }
};