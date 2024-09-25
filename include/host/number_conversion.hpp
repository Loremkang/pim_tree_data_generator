#pragma once

#include <cstdint>
#include <iostream>

class ConvertI64UI64 {
   public:
    static int64_t UI64ToI64(uint64_t ui64) {
        if (ui64 < (1ull << 63)) {
            return INT64_MIN + static_cast<int64_t>(ui64);
        } else {
            return static_cast<int64_t>(ui64 - (1ull << 63));
        }
    }

    // ret = i64 + (1ull << 63)
    static uint64_t I64ToUI64(int64_t i64) {
        if (i64 < 0) {
            return static_cast<uint64_t>(i64 - INT64_MIN);
        } else {
            return static_cast<uint64_t>(i64) + (1ull << 63);
        }
    }

    static bool IsCorrectConversion(int64_t i64, uint64_t ui64) {
        if (i64 < 0) {
            return ui64 == static_cast<uint64_t>(i64 - INT64_MIN);
        } else {
            if (ui64 < (1ull << 63)) {
                return false;
            }
            return (ui64 - (1ull << 63)) == static_cast<uint64_t>(i64);
        }
    }
};