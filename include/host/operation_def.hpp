#pragma once 
#include <cstdlib>

struct get_operation {
    int64_t key;
};

struct update_operation {
    int64_t key;
    int64_t value;
};

struct predecessor_operation {
    int64_t key;
};

struct insert_operation {
    int64_t key;
    int64_t value;
};

struct remove_operation {
    int64_t key;
};

struct scan_operation {
    int64_t lkey;
    int64_t rkey;
};

enum operation_t {
    empty_t = 0,
    get_t = 1,
    update_t = 2,
    predecessor_t = 3,
    insert_t = 4,
    remove_t = 5,
    scan_t = 6
};
const size_t kNumTypesOfOperations = 7;

struct union_operation {
    operation_t type;
    union {
        get_operation g;
        update_operation u;
        predecessor_operation p;
        scan_operation s;
        insert_operation i;
        remove_operation r;
    } tsk;
};