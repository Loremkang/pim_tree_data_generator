#pragma once

#include <iostream>
#include <cstdlib>
#include <parlay/utilities.h>

struct key_value {
    int64_t key;
    int64_t value;
    key_value(int64_t key_ = 0, int64_t value_ = 0) : key(key_), value(value_) {}
    key_value(std::pair<int64_t, int64_t> kv) : key(kv.first), value(kv.second) {}
    bool operator<(const key_value& kv) const {
        if (key > kv.key) return false;
        if (key < kv.key) return true;
        return (value < kv.value);
    }
    bool operator<=(const key_value& kv) const {
        return (*this < kv) || (*this == kv);
    }
    bool operator>(const key_value& kv) const {
        return !(*this <= kv);
    }
    bool operator>=(const key_value& kv) const {
        return !(*this < kv);
    }
    bool operator==(const key_value& kv) const {
        return (key == kv.key) && (value == kv.value);
    }
    bool operator!=(const key_value& kv) const {
        return !(*this == kv);
    }
    friend std::ostream& operator<<(std::ostream& os, const key_value& kv) {
        os << "{Key=" << kv.key << ", Value=" << kv.value << "}";
        return os;
    }

    static key_value default_kv_from_int64_key(int64_t key, int64_t dev) {
        int64_t new_value = static_cast<int64_t>(parlay::hash64(static_cast<uint64_t>(key + dev)));
        return key_value(key, new_value);
    }
};