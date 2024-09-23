#pragma once
#include <algorithm>
#include <parlay/primitives.h>
#include "operation_def.hpp"
using namespace std;
using namespace parlay;

namespace skew_generator {

class rn_gen {
   public:
    inline static vector<rn_gen> rand_gens;

    static int64_t randint64_rand() {
        uint64_t v = rand();
        v = (v << 31) + rand();
        v = (v << 2) + (rand() & 3);
        if (v >= (1ull << 63)) {
            return v - (1ull << 63);
        } else {
            return v - (1ull << 63);
        }
    }

    static void init() {
        srand(time(NULL));
        // srand(137);

        int thread_num = parlay::num_workers();
        cout<<"rand init: tn = " << thread_num << endl;
        for (int i = 0; i < thread_num; i++) {
            rand_gens.emplace_back(randint64_rand());
        }
    }

    static int64_t parallel_rand() {
        int tid = parlay::worker_id();
        return rand_gens[tid].next();
    }

    rn_gen(size_t seed) : state(seed){};
    rn_gen() : state(0){};
    int64_t random(int64_t i) { return parlay::hash64(this->state + i); }
    int64_t proceed(int64_t i) { this->state += i; }
    int64_t next() { return parlay::hash64(this->state++); }

   private:
    size_t state = 0;
};

double alpha = -1.0;
sequence<double> zipf_pos;

template <typename T>
void print_array(string name, T* arr, int l) {
    for (int i = 0; i < l; i++) {
        cout << name << "[" << i << "] = " << arr[i] << endl;
    }
}

void init_zipf_pos(int P) {
    double sum = 0;
    zipf_pos = sequence<double>(P, 1.0);
    for (int i = 0; i < P; i++) {
        zipf_pos[i] /= pow(i + 1, alpha);
        sum += zipf_pos[i];
    }
    for (int i = 0; i < P; i++) {
        zipf_pos[i] /= sum;
        if (i > 0) {
            zipf_pos[i] += zipf_pos[i - 1];
        }
    }
}

int biased_mapping(int64_t x, long siz, int bias, int l) {
    return min(siz - 1, l + abs(x) % siz / bias + 1);
}

void zipf_over_items(double a, slice<int*, int*> ids, int P, int ds_size) {
    if (a != alpha) {
        alpha = a;
        init_zipf_pos(P);
    }

    auto order = tabulate(P, [&](size_t i) { return i; });
    for (int i = 0; i < P; i++) {
        int r = abs(skew_generator::rn_gen::parallel_rand()) % (P - i);
        swap(order[i], order[i + r]);
    }

    auto slice_id = parlay::tabulate(ids.size(), [&](size_t i) {
        double rd = (double)(abs(skew_generator::rn_gen::parallel_rand())) / (double)INT64_MAX;
        int l = -1, r = P - 1;  // (]
        while (r - l > 1) {
            int mid = (l + r) >> 1;
            if (zipf_pos[mid] < rd) {
                l = mid;
            } else {
                r = mid;
            }
        }
        int position = r;
        return order[r];
    });

    auto ls = tabulate(P, [&](size_t i) { return i * (ds_size / P); });
    parallel_for(0, ids.size(), [&](size_t i) {
        int64_t x = skew_generator::rn_gen::parallel_rand();
        ids[i] = biased_mapping(x, ds_size, P, ls[slice_id[i]]);
    });
}

void all_or_nothing_items(int bias, slice<int*, int*> ids, int ds_size) {
    int64_t l = (abs(skew_generator::rn_gen::parallel_rand()) % bias) * (ds_size / bias);
    parallel_for(0, ids.size(), [&](size_t i) {
        int64_t x = skew_generator::rn_gen::parallel_rand();
        ids[i] = biased_mapping(x, ds_size, bias, l);
    });
}

void zipf_over_keys(double a, slice<int64_t*, int64_t*> keys, int P) {
    if (a != alpha) {
        alpha = a;
        init_zipf_pos(P);
    }

    auto order = tabulate(P, [&](size_t i) { return i; });
    for (int i = 0; i < P; i++) {
        int r = abs(skew_generator::rn_gen::parallel_rand()) % (P - i);
        swap(order[i], order[i + r]);
    }

    auto slice_id = parlay::tabulate(keys.size(), [&](size_t i) {
        double rd = (double)(abs(skew_generator::rn_gen::parallel_rand())) / (double)INT64_MAX;
        int l = -1, r = P - 1;  // (]
        while (r - l > 1) {
            int mid = (l + r) >> 1;
            if (zipf_pos[mid] < rd) {
                l = mid;
            } else {
                r = mid;
            }
        }
        int position = r;
        return order[r];
    });

    int64_t piece_size = (UINT64_MAX / P);
    auto ls = tabulate(P, [&](int64_t i) {
        return INT64_MIN + piece_size / 2 + piece_size * i;
    });

    parallel_for(0, keys.size(), [&](size_t i) {
        int64_t x = skew_generator::rn_gen::parallel_rand() / P + ls[slice_id[i]];
        keys[i] = x;
    });
}

void all_or_nothing_keys(int bias, slice<int64_t*, int64_t*> ids) {
    uint64_t piece_size = (UINT64_MAX / bias);
    int64_t piece_id = abs(skew_generator::rn_gen::parallel_rand()) % bias;
    int64_t l = INT64_MIN + piece_size / 2 + piece_size * piece_id;
    parallel_for(0, ids.size(), [&](size_t i) {
        int64_t x = skew_generator::rn_gen::parallel_rand() / bias + l;
        ids[i] = x;
    });
}

}  // namespace skew_generator

class test_generator {
   public:
    sequence<double> possibilities = sequence<double>(OPERATION_NR_ITEMS);
    int batch_size;

    test_generator(slice<double*, double*> _pos, int _size) {
        assert(_pos[0] == 0.0);
        possibilities[0] = 0.0;
        for (int i = 1; i < OPERATION_NR_ITEMS; i++) {
            possibilities[i] = possibilities[i - 1] + _pos[i];
        }
        assert(possibilities[OPERATION_NR_ITEMS - 1] == 1.0);
        batch_size = _size;
    }

    void fill(operation& op, int64_t& k, int64_t& v, int64_t& in_key,
              double& rd) {
        for (int T = 0; T < OPERATION_NR_ITEMS; T++) {
            if (rd < possibilities[T]) {
                if (T == 1) {
                    op.type = operation_t::get_t;
                    op.tsk.g = (get_operation){.key = in_key};
                } else if (T == 2) {
                    op.type = operation_t::update_t;
                    op.tsk.u = (update_operation){.key = in_key, .value = v};
                } else if (T == 3) {
                    op.type = operation_t::predecessor_t;
                    op.tsk.p = (predecessor_operation){.key = k};
                } else if (T == 4) {
                    op.type = operation_t::scan_t;
                    if (k > v) swap(k, v);
                    op.tsk.s = (scan_operation){.lkey = k, .rkey = v};
                } else if (T == 5) {
                    op.type = operation_t::insert_t;
                    op.tsk.i = (insert_operation){.key = k, .value = v};
                } else if (T == 6) {
                    op.type = operation_t::remove_t;
                    op.tsk.r = (remove_operation){.key = in_key};
                } else {
                    assert(false);
                }
                assert(op.type == T);
                break;
            }
        }
    }
    void fill_with_random_ops(slice<operation*, operation*> ops) {
        int n = ops.size();
        parlay::parallel_for(0, n, [&](size_t i) {
            int64_t k = skew_generator::rn_gen::parallel_rand();
            int64_t v = skew_generator::rn_gen::parallel_rand();
            double rd = (double(abs(skew_generator::rn_gen::parallel_rand()) % 16384)) / 16384;
            int64_t in_key = 0;
            fill(ops[i], k, v, in_key, rd);
        });
    }

    void fill_with_get_ops(slice<operation*, operation*> ops, bool zipf,
                           double alpha, int bias,
                           batch_parallel_oracle& oracle, int batch_size) {
        int n = ops.size();
        auto&& keys = oracle.dump();
        assert((n % batch_size) == 0);
        auto ids = parlay::sequence<int>(batch_size);
        for (int i = 0; i < n; i += batch_size) {
            auto sub_slice = ops.cut(i, i + batch_size);
            if (zipf) {
                skew_generator::zipf_over_items(alpha, make_slice(ids),
                                                nr_of_dpus, keys.size());
            } else {
                skew_generator::all_or_nothing_items(bias, make_slice(ids),
                                                     keys.size());
            }
            parlay::parallel_for(0, batch_size, [&](size_t i) {
                operation op;
                op.type = operation_t::get_t;
                op.tsk.g.key = keys[ids[i]].key;
                sub_slice[i] = op;
            });
        }
    }

    void fill_with_update_ops(slice<operation*, operation*> ops, bool zipf,
                              double alpha, int bias,
                              batch_parallel_oracle& oracle, int batch_size) {
        int n = ops.size();
        auto&& keys = oracle.dump();
        assert((n % batch_size) == 0);
        auto ids = parlay::sequence<int>(batch_size);
        for (int i = 0; i < n; i += batch_size) {
            auto sub_slice = ops.cut(i, i + batch_size);
            if (zipf) {
                skew_generator::zipf_over_items(alpha, make_slice(ids),
                                                nr_of_dpus, keys.size());
            } else {
                skew_generator::all_or_nothing_items(bias, make_slice(ids),
                                                     keys.size());
            }
            parlay::parallel_for(0, batch_size, [&](size_t i) {
                operation op;
                op.type = operation_t::update_t;
                op.tsk.u.key = keys[ids[i]].key;
                op.tsk.u.value = skew_generator::rn_gen::parallel_rand();
                sub_slice[i] = op;
            });
        }
    }

    void fill_with_search_ops(slice<operation*, operation*> ops, bool zipf,
                              double alpha, int bias,
                              batch_parallel_oracle& oracle, int batch_size) {
        int n = ops.size();
        assert((n % batch_size) == 0);
        auto ids = parlay::sequence<int64_t>(batch_size);
        for (int i = 0; i < n; i += batch_size) {
            auto sub_slice = ops.cut(i, i + batch_size);
            if (zipf) {
                skew_generator::zipf_over_keys(alpha, make_slice(ids),
                                               nr_of_dpus);
            } else {
                skew_generator::all_or_nothing_keys(bias, make_slice(ids));
            }
            parlay::parallel_for(0, batch_size, [&](size_t i) {
                operation op;
                op.type = operation_t::predecessor_t;
                op.tsk.p.key = ids[i];
                sub_slice[i] = op;
            });
        }
    }

    void fill_with_insert_ops(slice<operation*, operation*> ops, bool zipf,
                              double alpha, int bias,
                              batch_parallel_oracle& oracle, int batch_size) {
        int n = ops.size();
        assert((n % batch_size) == 0);
        auto ids = parlay::sequence<int64_t>(batch_size);
        for (int i = 0; i < n; i += batch_size) {
            auto sub_slice = ops.cut(i, i + batch_size);
            if (zipf) {
                skew_generator::zipf_over_keys(alpha, make_slice(ids),
                                               nr_of_dpus);
            } else {
                skew_generator::all_or_nothing_keys(bias, make_slice(ids));
            }

            parlay::parallel_for(0, batch_size, [&](size_t i) {
                operation op;
                op.type = operation_t::insert_t;
                op.tsk.i.key = ids[i];
                op.tsk.i.value = skew_generator::rn_gen::parallel_rand();
                sub_slice[i] = op;
            });
        }
    }

    void fill_with_remove_ops(slice<operation*, operation*> ops, bool zipf,
                              double alpha, int bias,
                              batch_parallel_oracle& oracle, int batch_size) {
        int n = ops.size();
        assert((n % batch_size) == 0);
        auto ids = parlay::sequence<int>(batch_size);
        for (int i = 0; i < n; i += batch_size) {
            auto sub_slice = ops.cut(i, i + batch_size);
            if (zipf) {
                skew_generator::zipf_over_items(
                    alpha, make_slice(ids), nr_of_dpus, oracle.size());
            } else {
                skew_generator::all_or_nothing_items(bias, make_slice(ids),
                                                     oracle.size());
            }
            parlay::parallel_for(0, batch_size, [&](size_t i) {
                operation op;
                op.type = operation_t::remove_t;
                op.tsk.r.key = oracle.inserted[ids[i]].key;
                sub_slice[i] = op;
            });
            auto toremove = parlay::delayed_seq<int64_t>(
                batch_size, [&](size_t i) { return sub_slice[i].tsk.r.key; });
            oracle.remove_batch(make_slice(toremove));
        }
    }
    
    // Range Scan
    void fill_with_scan_ops(slice<operation*, operation*> ops, bool zipf,
                            double alpha, int bias,
                            batch_parallel_oracle& oracle, int batch_size,
                            int expected_length = 100) {
        int n = ops.size();
        assert((n % batch_size) == 0);
        auto ids = parlay::sequence<int64_t>(batch_size);
        for (int i = 0; i < n; i += batch_size) {
            auto sub_slice = ops.cut(i, i + batch_size);
            if (zipf) {
                skew_generator::zipf_over_keys(alpha, make_slice(ids),
                                               nr_of_dpus);
            } else {
                skew_generator::all_or_nothing_keys(bias, make_slice(ids));
            }
            parlay::parallel_for(0, batch_size, [&](size_t i) {
                operation op;
                op.type = operation_t::scan_t;
                op.tsk.s.lkey = ids[i];
                op.tsk.s.rkey = ids[i] +
                    (UINT64_MAX / oracle.size() * expected_length);
                assert(expected_length == 100);
                sub_slice[i] = op;
            });
        }
    }

    void fill_with_biased_ops(slice<operation*, operation*> ops, bool zipf,
                              double alpha, int bias,
                              batch_parallel_oracle& oracle, int batch_size) {
        int n = ops.size();
        assert((n % batch_size) == 0);
        bool single_type = true;
        for (int t = 0; t < OPERATION_NR_ITEMS; t ++) {
            if (possibilities[t] != 1.0 && possibilities[t] != 0.0) {
                single_type = false;
                break;
            }
        }
        printf("single type=%d\n", single_type);
        if (single_type) { // micro benchmarks
            double rd = (double)(abs(skew_generator::rn_gen::parallel_rand())) / (double)INT64_MAX;
            for (int t = 0; t < OPERATION_NR_ITEMS; t++) {
                if (rd < possibilities[t]) {
                // if (possibilities[t] == 1.0) {
                    if (t == 1) {
                        fill_with_get_ops(ops, zipf, alpha, bias, oracle,
                                          batch_size);
                    } else if (t == 2) {
                        assert(false);
                    } else if (t == 3) {
                        fill_with_search_ops(ops, zipf, alpha, bias, oracle,
                                             batch_size);
                    } else if (t == 4) {
                        fill_with_scan_ops(ops, zipf, alpha, bias, oracle,
                                           batch_size);
                    } else if (t == 5) {
                        fill_with_insert_ops(ops, zipf, alpha, bias, oracle,
                                             batch_size);
                    } else if (t == 6) {
                        auto oracle2 = oracle;
                        cout<<"remove: init size: "<<oracle2.size()<<endl;
                        fill_with_remove_ops(ops, zipf, alpha, bias, oracle2,
                                             batch_size);
                        cout<<"remove: final size: "<<oracle2.size()<<endl;
                    } else {
                        assert(false);
                    }
                    break;
                }
            }
        } else { // used only for debug
            auto&& kvs = oracle.dump();
            int m = kvs.size();
            auto keys = parlay::delayed_seq<int64_t>(m, [&](size_t i) {
                return kvs[i].key;
            });
            parlay::parallel_for(0, n, [&](size_t i) {
                int64_t k = skew_generator::rn_gen::parallel_rand() / bias;
                int64_t v = skew_generator::rn_gen::parallel_rand();
                double rd =
                    (double(abs(skew_generator::rn_gen::parallel_rand()) % 16384)) / 16384;
                int64_t in_key = 0;
                if (m > 0) {
                    in_key = abs(k) % m + 1;
                    in_key = keys[in_key];
                }
                fill(ops[i], k, v, in_key, rd);
            });
        }
    }
};
