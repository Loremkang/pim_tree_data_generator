#pragma once

#include <iostream>
#include <cstdlib>

struct key_value {
    uint64_t key;
    uint64_t value;
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
};