#pragma once

#include <argparse/argparse.hpp>
#include <parlay/slice.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <string>
#include <shared_mutex>
#include "fcntl.h"
#include "oracle.hpp"
#include "operation_def.hpp"
#include "timer.hpp"
#include "test_generator.hpp"
#include "operation.hpp"
#include "parlay/papi/papi_util_impl.h"
#include <unistd.h>
#include <sys/stat.h>
#include <filesystem>

using namespace std;
using namespace parlay;
namespace fs = std::filesystem;

void dpu_energy_stats(bool flag = false) {
    #ifdef DPU_ENERGY
        uint64_t db_iter=0, op_iter=0, cycle_iter=0, instr_iter=0;
        uint64_t op_total = 0, db_size_total = 0, cycle_total = 0;
        DPU_FOREACH(dpu_set, dpu, each_dpu) {
            DPU_ASSERT(dpu_copy_from(dpu, "op_count", 0, &op_iter, sizeof(uint64_t)));
            DPU_ASSERT(dpu_copy_from(dpu, "db_size_count", 0, &db_iter, sizeof(uint64_t)));
            DPU_ASSERT(dpu_copy_from(dpu, "cycle_count", 0, &cycle_iter, sizeof(uint64_t)));
            op_total += op_iter;
            db_size_total += db_iter;
            cycle_total += cycle_iter;
            if(flag) {
                cout<<"DPU ID: "<<each_dpu<<" "
                    <<op_iter<<" "<<db_iter<<" ";
                cout<<((op_iter > 0) ? (db_iter / op_iter) : 0)
                    <<" "<<cycle_iter<<endl;
            }
        }
        cout<<"op_total: "<<op_total<<endl;
        cout<<"db_total: "<<db_size_total<<endl;
        cout<<"cy_total: "<<cycle_total<<endl;
    #endif
}

inline void write_ops_to_file(string file_name,
                              slice<operation*, operation*> ops) {
    printf("Will write to '%s'\n", file_name.c_str());

    /* Open a file for writing.
     *  - Creating the file if it doesn't exist.
     *  - Truncating it to 0 size if it already exists. (not really needed)
     *
     * Note: "O_WRONLY" mode is not sufficient when mmaping.
     */

    const char* filepath = file_name.c_str();

    int try_to_unlink = unlink(filepath);

    if (try_to_unlink == 0) {
        cout << "Remove: " << filepath << endl;
    }

    int fd = open(filepath, O_RDWR | O_CREAT | O_TRUNC, (mode_t)0600);

    if (fd == -1) {
        perror("Error opening file for writing");
        exit(EXIT_FAILURE);
    }

    // Stretch the file size to the size of the (mmapped) array of char

    int n = ops.size();
    size_t filesize = sizeof(operation) * n;

    if (lseek(fd, filesize - sizeof(operation), SEEK_SET) == -1) {
        close(fd);
        perror("Error calling lseek() to 'stretch' the file");
        exit(EXIT_FAILURE);
    }

    operation empty_op;
    empty_op.type = operation_t::empty_t;

    if (write(fd, &empty_op, sizeof(operation)) == -1) {
        close(fd);
        perror("Error writing last byte of the file");
        exit(EXIT_FAILURE);
    }

    // Now the file is ready to be mmapped.
    void* map = mmap(0, filesize, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (map == MAP_FAILED) {
        close(fd);
        perror("Error mmapping the file");
        exit(EXIT_FAILURE);
    }

    operation* fops = (operation*)map;

    parlay::parallel_for(0, n, [&](size_t i) { fops[i] = ops[i]; });

    cout << "task generation finished" << endl;

    // Write it now to disk
    if (msync(map, filesize, MS_SYNC) == -1) {
        perror("Could not sync the file to disk");
    }

    // Don't forget to free the mmapped memory
    if (munmap(map, filesize) == -1) {
        close(fd);
        perror("Error un-mmapping the file");
        exit(EXIT_FAILURE);
    }

    // Un-mmaping doesn't close the file, so we still need to do that.
    close(fd);
}

bool Check_result = false;

template <typename Checker>
auto read_op_file(string name, Checker checker) {
    const char* filepath = name.c_str();

    int fd = open(filepath, O_RDONLY, (mode_t)0600);

    if (fd == -1) {
        perror("Error opening file for writing");
        exit(EXIT_FAILURE);
    }

    struct stat fileInfo;

    if (fstat(fd, &fileInfo) == -1) {
        perror("Error getting the file size");
        exit(EXIT_FAILURE);
    }

    if (fileInfo.st_size == 0) {
        fprintf(stderr, "Error: File is empty, nothing to do\n");
        exit(EXIT_FAILURE);
    }

    printf("File size is %ji\n", (intmax_t)fileInfo.st_size);

    void* map = mmap(0, fileInfo.st_size, PROT_READ, MAP_SHARED, fd, 0);

    if (map == MAP_FAILED) {
        close(fd);
        perror("Error mmapping the file");
        exit(EXIT_FAILURE);
    }

    cout << fileInfo.st_size << ' ' << sizeof(operation) << endl;

    assert(fileInfo.st_size % sizeof(operation) == 0);

    int n = fileInfo.st_size / sizeof(operation);
    auto operation_map = (operation*)map;
    auto operations = parlay::tabulate(n, [&](size_t i) {
        operation& op = operation_map[i];
        checker(op);
        return op;
    });

    return operations;
}

namespace core {

batch_parallel_oracle oracle;
bool check_result = true;
atomic<int> batch_number = 0;

int num_top_level_threads;
int push_pull_limit_dynamic;

shared_mutex op_mutex;

void get(slice<get_operation*, get_operation*> ops, unique_lock<mutex>& mut, int tid = 0) {
    parlay::sequence<int64_t> ops_sequence;
    pim_skip_list* ds = &pim_skip_list_drivers[tid];
    int n = ops.size();
    {
        if (check_result) {
            ops_sequence =
                parlay::tabulate(n, [&](size_t i) { return ops[i].key; });
        }
        auto ops2 = make_slice((int64_t*)ops.begin(), (int64_t*)ops.end());
        time_nested("get load", [&]() { ds->get_load(ops2); });
        mut.unlock();
    }

    {
        shared_lock rLock(op_mutex);
        cout << (batch_number++) << " " << __FUNCTION__ << " " << tid << endl;
        time_nested("get", [&]() {
            ds->get();
        });
        if (check_result) {
            auto v1 = parlay::make_slice(ds->kv_output, ds->kv_output + n);
            auto pred = oracle.predecessor_batch(make_slice(ops_sequence));
            auto v2 = parlay::tabulate(pred.size(), [&](size_t i) {
                if (pred[i].key != ops_sequence[i]) {
                    return (key_value){.key = INT64_MIN, .value = INT64_MIN};
                } else {
                    return pred[i];
                }
            });
            if (!parlay::equal(v1, v2)) {
                int n = v1.size();
                for (int i = 0; i < n; i++) {
                    if (v1[i] != v2[i]) {
                        printf("[%8d]\t", i);
                        cout << "k=" << ops[i].key << "\tv1=" << v1[i]
                             << "\tv2=" << v2[i] << endl;
                    }
                }
            }
        }
    }
}

void update(slice<update_operation*, update_operation*> ops,
            unique_lock<mutex>& mut, int tid = 0) {
    assert(false);
}

// Range Scan
void scan(slice<scan_operation*, scan_operation*> ops, unique_lock<mutex>& mut, int tid = 0,
          bool reset_len=false, int64_t expected_length = 100, uint64_t dataset_size=500000000) {
    pim_skip_list* ds = &pim_skip_list_drivers[tid];
    if(reset_len) {
        int64_t range_size;
        if(check_result)
            range_size = UINT64_MAX / oracle.size() * expected_length;
        else
            range_size = UINT64_MAX / dataset_size * expected_length;
        parfor_wrap(0, ops.size(), [&](size_t i) {
            if (ops[i].lkey < INT64_MAX - range_size)
                ops[i].rkey = ops[i].lkey + range_size;
        });
    }
    parfor_wrap(0, ops.size(), [&](size_t i) {
        if (ops[i].lkey >= ops[i].rkey)
            ops[i].rkey = ops[i].lkey + 1;
    });
    // slice<scan_operation*, scan_operation*> ops2 = ops;
    mut.unlock();
    shared_lock rLock(op_mutex);
    cout << (batch_number++) << " " << __FUNCTION__ << " " << tid << endl;
    time_start("scan");
    auto v1 = ds->scan(ops);
    time_end("scan");
    if (Check_result) {
        int64_t length = ops.size();
        auto v2 = oracle.scan_size_batch(ops);
        bool correct = true;
        parlay::parallel_for(0, length, [&](size_t i) {
            if (correct) {
                if ((v2[i].second - v2[i].first) !=
                    (v1.second[i].second - v1.second[i].first)) {
                    correct = false;
                }
            }
        });
        if (!correct) {
            int64_t v1l, v2l, res = 0, print_num = 0;
            for (int64_t i = 0; i < length; i++) {
                v1l = v1.second[i].second - v1.second[i].first;
                v2l = v2[i].second - v2[i].first;
                if (v2l != v1l) {
                    if(print_num < 10){
                        printf(
                            "k=(%lld,%lld) v1_s=%lld v2_s=%lld (%lld, %lld; %lld, %lld) (%lld, %lld; %lld, %lld)\n",
                            ops[i].lkey, ops[i].rkey, v1l, v2l,
                            v1.first[v1.second[i].first - 1].key, v1.first[v1.second[i].first].key,
                            v1.first[v1.second[i].second].key, v1.first[v1.second[i].second + 1].key,
                            oracle.inserted[v2[i].first - 1].key, oracle.inserted[v2[i].first].key,
                            oracle.inserted[v2[i].second].key, oracle.inserted[v2[i].second + 1].key
                        );
                        print_num++;
                    }
                    res++;
                }
            }
            cout << "Number of Errorness: " << res << endl;
            cout << "kv_set size returned: " << v1.first.size() << endl;
        }
    }
}

void predecessor(slice<predecessor_operation*, predecessor_operation*> ops,
                 unique_lock<mutex>& mut, int tid = 0) {
    parlay::sequence<int64_t> ops_sequence;
    int n = ops.size();
    pim_skip_list* ds = &pim_skip_list_drivers[tid];
    {
        if (check_result) {
            ops_sequence =
                parlay::tabulate(n, [&](size_t i) { return ops[i].key; });
        }
        auto ops2 = make_slice((int64_t*)ops.begin(), (int64_t*)ops.end());
        time_nested("predecessor load", [&]() { ds->predecessor_load(ops2); });
        mut.unlock();
    }
    {
        shared_lock rLock(op_mutex);
        // cout << (batch_number++) << " " << __FUNCTION__ << " " << tid << ' ' << ops.size() << endl;

        time_nested("predecessor", [&]() { ds->predecessor(); });

        if (check_result) {
            auto v1 = parlay::make_slice(ds->kv_output, ds->kv_output + n);
            auto v2 = oracle.predecessor_batch(make_slice(ops_sequence));
            if (!parlay::equal(v1, v2)) {
                int n = v1.size();
                for (int i = 0; i < n; i++) {
                    if (v1[i] != v2[i]) {
                        printf("[%8d]\t", i);
                        cout << "k=" << ops[i].key << "\tv1=" << v1[i]
                             << "\tv2=" << v2[i] << endl;
                    }
                }
            }
        }
    }
}

void insert(slice<insert_operation*, insert_operation*> ops,
            unique_lock<mutex>& mut, int tid = 0) {
    parlay::sequence<key_value> ops_sequence;
    int n = ops.size();
    pim_skip_list* ds = &pim_skip_list_drivers[tid];
    {
        if (check_result) {
            ops_sequence =
                parlay::tabulate(n, [&](size_t i) { return (key_value){.key = ops[i].key, .value = ops[i].value}; });
        }
        auto ops2 = make_slice((key_value*)ops.begin(), (key_value*)ops.end());
        time_nested("insert load", [&]() { ds->insert_load(ops2); });
        mut.unlock();
    }
    {
        unique_lock wLock(op_mutex);
        cout << (batch_number++) << " " << __FUNCTION__ << " " << tid << " " << n << endl;
        time_nested("insert", [&]() { ds->insert(); });
        if (check_result) {
            oracle.insert_batch(make_slice(ops_sequence));
        }
    }
}

void remove(slice<remove_operation*, remove_operation*> ops,
            unique_lock<mutex>& mut, int tid = 0) {
    parlay::sequence<int64_t> ops_sequence;
    int n = ops.size();
    pim_skip_list* ds = &pim_skip_list_drivers[tid];
    {
        if (check_result) {
            ops_sequence =
                parlay::tabulate(n, [&](size_t i) { return ops[i].key; });
        }
        auto ops2 = make_slice((int64_t*)ops.begin(), (int64_t*)ops.end());
        time_nested("remove load", [&]() { ds->remove_load(ops2); });
        mut.unlock();
    }
    {
        unique_lock wLock(op_mutex);
        cout << (batch_number++) << " " << __FUNCTION__ << " " << tid  << " " << n << endl;
        time_nested("remove", [&]() { ds->remove(); });
        if (check_result) {
            oracle.remove_batch(make_slice(ops_sequence));
        }
    }
}

const int _block_size = 1000;
int l;
parlay::sequence<size_t>* sums;
int cnts[OPERATION_NR_ITEMS];
int n, rounds;
int T;

void init(parlay::slice<operation*, operation*> ops, int load_batch_size,
          int execute_batch_size) {
    memset(op_count, 0, sizeof(op_count));
    l = parlay::internal::num_blocks(load_batch_size, _block_size);
    sums = new parlay::sequence<size_t>[OPERATION_NR_ITEMS];
    for (int i = 0; i < OPERATION_NR_ITEMS; i++) {
        sums[i] = parlay::sequence<size_t>(l);
        cnts[i] = 0;
    }
    n = ops.size();
    rounds = parlay::internal::num_blocks(n, load_batch_size);
    T = 0;
}

mutex load_batch_mutex;

bool load_one_batch(parlay::slice<operation*, operation*> ops,
                    int load_batch_size) {
    if (T >= rounds) {
        return false;
    }
    int l = T * load_batch_size;
    int r = min((T + 1) * load_batch_size, n);
    int len = r - l;

    auto mixed_op_batch = ops.cut(l, r);

    parlay::internal::sliced_for(
        len, _block_size, [&](size_t i, size_t s, size_t e) {
            size_t c[OPERATION_NR_ITEMS] = {0};
            for (size_t j = s; j < e; j++) {
                int t = mixed_op_batch[j].type;
                assert(j < (size_t)len);
                assert(t >= 0 && t < OPERATION_NR_ITEMS);
                c[t]++;
            }
            for (int j = 0; j < OPERATION_NR_ITEMS; j++) {
                sums[j][i] = c[j];
            }
        });
    for (int j = 0; j < OPERATION_NR_ITEMS; j++) {
        cnts[j] = parlay::scan_inplace(parlay::make_slice(sums[j]),
                                       parlay::addm<size_t>());
    }
    parlay::internal::sliced_for(
        len, _block_size, [&](size_t i, size_t s, size_t e) {
            size_t c[OPERATION_NR_ITEMS];
            for (int j = 0; j < OPERATION_NR_ITEMS; j++) {
                c[j] = sums[j][i] + op_count[j];
            }
            for (size_t j = s; j < e; j++) {
                
                operation t = mixed_op_batch[j];
                
                operation_t& operation_type = t.type;
                int x = (int)operation_type;
                switch (operation_type) {
                    case operation_t::get_t: {
                        get_ops[c[x]++] = t.tsk.g;
                        break;
                    }
                    case operation_t::update_t: {
                        update_ops[c[x]++] = t.tsk.u;
                        break;
                    }
                    case operation_t::predecessor_t: {
                        predecessor_ops[c[x]++] = t.tsk.p;
                        break;
                    }
                    case operation_t::scan_t: {
                        scan_ops[c[x]++] = t.tsk.s;
                        break;
                    }
                    case operation_t::insert_t: {
                        insert_ops[c[x]++] = t.tsk.i;
                        break;
                    }
                    case operation_t::remove_t: {
                        remove_ops[c[x]++] = t.tsk.r;
                        break;
                    }
                    default: {
                        assert(false);
                    }
                }
            }
        });
    for (int j = 0; j < OPERATION_NR_ITEMS; j++) {
        op_count[j] += cnts[j];
    }
    T++;
    return true;
}

operation_t batch_ready(int execute_batch_size) {
    int scan_execute_batch_size = execute_batch_size / 100;
    for (int j = 1; j < OPERATION_NR_ITEMS; j++) {
        if (op_count[j] >= execute_batch_size && op_count[j] > 0) {
            return (operation_t)j;
        }
    }
    if (op_count[operation_t::scan_t] >= scan_execute_batch_size && op_count[operation_t::scan_t] > 0) {
        return operation_t::scan_t;
    }
    return operation_t::empty_t;
}

int scan_start = 0;

void run_batch(operation_t op_type, unique_lock<mutex>& mut, int tid) {
    int count = op_count[(int)op_type];
    if(op_type != operation_t::scan_t)
        op_count[(int)op_type] = 0;

    switch (op_type) {
        case operation_t::get_t: {
            core::get(parlay::make_slice(get_ops, get_ops + count), mut, tid);
            break;
        }
        case operation_t::update_t: {
            assert(false);
            break;
        }
        case operation_t::predecessor_t: {
            core::predecessor(
                parlay::make_slice(predecessor_ops, predecessor_ops + count),
                mut, tid);
            break;
        }
        case operation_t::scan_t: {
            int scan_batch = 10000;
            if (count - scan_start >= scan_batch) {
                core::scan(parlay::make_slice(scan_ops + scan_start, scan_ops + scan_start + scan_batch), mut, tid);
                scan_start += scan_batch;
            } else if(count - scan_start > 1) {
                parlay::parallel_for(0, count - scan_start, [&](size_t i) {
                    scan_ops[i] = scan_ops[i + scan_start];
                });
                op_count[(int)op_type] = count - scan_start;
                scan_start = 0;
                core::scan(parlay::make_slice(scan_ops, scan_ops + op_count[(int)op_type]), mut, tid);
                op_count[(int)op_type] = 0;
            }
            else {
                op_count[(int)op_type] = 0;
                scan_start = 0;
                mut.unlock();
            }
            break;
        }
        case operation_t::insert_t: {
            core::insert(parlay::make_slice(insert_ops, insert_ops + count), mut, tid);
            break;
        }
        case operation_t::remove_t: {
            core::remove(parlay::make_slice(remove_ops, remove_ops + count), mut, tid);
            break;
        }
        default: {
            assert(false);
            break;
        }
    }
}

bool finished() {
    if (T < rounds) {
        return false;
    }
    return true;
}

void execute(parlay::slice<operation*, operation*> ops, int load_batch_size,
             int execute_batch_size, int threads) {
    printf("execute n=%lu batchsize=%d,%d\n", ops.size(), load_batch_size,
           execute_batch_size);
    ASSERT(threads <= num_top_level_threads);
    memset(op_count, 0, sizeof(op_count));
    init(ops, load_batch_size, execute_batch_size);
    atomic<int> num_finished_threads = 0;
    parlay::parallel_for(
        0, threads,
        [&](size_t tid) {
            cpu_coverage_timer->start();
            pim_coverage_timer->start();
            pim_coverage_timer->end();
            time_nested("global_exec", [&]() {
                printf("%d / %d *****!!! start\n", tid, threads);
                while (true) {
                    {
                        unique_lock<mutex> lock(load_batch_mutex);
                        time_start("load batch");
                        operation_t op_type = operation_t::empty_t;
                        while (true) {
                            op_type = batch_ready(execute_batch_size);
                            if (op_type != operation_t::empty_t) break;
                            bool next_batch =
                                load_one_batch(ops, load_batch_size);
                            if (!next_batch) {
                                op_type = batch_ready(1); // finish remaining tasks
                                break;
                            }
                        }
                        time_end("load batch");

                        if (op_type == operation_t::empty_t) {
                            break; // !next_batch
                        }
                        run_batch(op_type, lock, tid);  // may unlock here
                    }
                }
                cout << tid << "*****!!! finished" << endl;
                num_finished_threads++;
                if (tid == 0) {
                    while (num_finished_threads.load() < threads) {
                        this_thread::sleep_for(chrono::microseconds(100));
                    }
                }
            });
            cpu_coverage_timer->end();
            pim_coverage_timer->start();
            pim_coverage_timer->end();
            if (tid == 0) {
                assert(num_finished_threads.load() == threads);
            }
        },
        1);
    printf("execute finish!\n");
    fflush(stdout);
    cout << oracle.size() << endl;
}

};  // namespace core

class frontend {
   public:
    virtual sequence<operation> init_tasks() = 0;
    virtual sequence<operation> test_tasks() = 0;
};

class frontend_by_file : public frontend {
   public:
    string init_file;
    string test_file;
    int init_n;
    int test_n;

    frontend_by_file(string _if, string _tf, int _in = -1, int _tn = -1) {
        init_file = _if;
        test_file = _tf;
        init_n = _in;
        test_n = _tn;
    }

    sequence<operation> init_tasks() {
        auto ops = read_op_file(init_file, [&](const operation& op) {
            assert(op.type == insert_t);
        });
        if (init_n == -1) {
            return ops;
        } else {
            return tabulate(init_n, [&](size_t i) { return ops[i]; });
        }
    }

    sequence<operation> test_tasks() {
        auto ops = read_op_file(test_file, [&](const operation& op) {
            assert(op.type != empty_t);
        });
        if (test_n == -1) {
            return ops;
        } else {
            return tabulate(test_n, [&](size_t i) { return ops[i]; });
        }
    }
};

class frontend_by_generation : public frontend {
   public:
    int init_n, test_n;
    sequence<double> pos;
    int bias;
    batch_parallel_oracle oracle;
    int init_batch_size;
    int test_batch_size;

    frontend_by_generation(int _init_n, int _test_n, sequence<double> _pos,
                           int _bias, int _init_batch_size, int _test_batch_size)
        : init_n{_init_n},
          test_n{_test_n},
          pos{_pos},
          bias{_bias},
          init_batch_size{_init_batch_size},
          test_batch_size{_test_batch_size} {}

    sequence<operation> init_tasks() {
        sequence<double> init_pos = sequence<double>(OPERATION_NR_ITEMS, 0);
        init_pos[operation_t::insert_t] = 1.0;
        test_generator tg(make_slice(init_pos), init_batch_size);
        auto ops = sequence<operation>(init_n);
        tg.fill_with_random_ops(make_slice(ops));
        auto kvs = parlay::delayed_seq<key_value>(ops.size(), [&](size_t i) {
            return (key_value){.key = ops[i].tsk.i.key,
                               .value = ops[i].tsk.i.value};
        });
        oracle.init(make_slice(kvs));
        return ops;
    }

    sequence<operation> test_tasks() {
        assert(pos.size() == OPERATION_NR_ITEMS);
        test_generator tg(make_slice(this->pos), test_batch_size);
        auto ops = sequence<operation>(test_n);
        tg.fill_with_biased_ops(make_slice(ops), false, 0.0, bias, oracle,
                                test_batch_size);
        return ops;
    }
};

class frontend_testgen {
   public:
    int init_n, test_n;
    sequence<double> pos;
    int bias;
    string init_file;
    string test_file;
    batch_parallel_oracle oracle;
    int execute_batch_size;

    frontend_testgen(int _init_n, int _test_n, sequence<double> _pos, int _bias,
                     string initfile, string testfile, int batch_size)
        : init_n{_init_n},
          test_n{_test_n},
          pos{_pos},
          bias{_bias},
          init_file{initfile},
          test_file{testfile},
          execute_batch_size{batch_size} {}

    sequence<operation> generate_tasks(parlay::sequence<double>& possi, int n,
                                       bool zipf, double alpha, int bias) {
        assert(pos.size() == OPERATION_NR_ITEMS);
        test_generator tg(make_slice(possi), execute_batch_size);
        auto ops = sequence<operation>(n);
        if(pos[operation_t::scan_t] > 0) {
            tg.fill_with_biased_ops(make_slice(ops), zipf, alpha, bias, oracle,
                                    execute_batch_size / 100);
        } else {
            tg.fill_with_biased_ops(make_slice(ops), zipf, alpha, bias, oracle,
                                    execute_batch_size);
        }
        return ops;
    }

    void init_oracle(slice<operation*, operation*> ops) {
        auto kvs = parlay::delayed_seq<key_value>(ops.size(), [&](size_t i) {
            return (key_value){.key = ops[i].tsk.i.key,
                               .value = ops[i].tsk.i.value};
        });
        oracle.init(make_slice(kvs));
    }

    void write_init_file(string init_file_name) {
        auto init_pos = sequence<double>(OPERATION_NR_ITEMS, 0);
        init_pos[operation_t::insert_t] = 1.0;
        auto ops = generate_tasks(init_pos, init_n, false, 0.0, 1);
        init_oracle(make_slice(ops));
        write_ops_to_file(init_file_name, make_slice(ops));
        auto ops_sorted =
            parlay::sort(ops, [](const operation& a, const operation& b) {
                return a.tsk.i.key < b.tsk.i.key;
            });
        write_ops_to_file(init_file_name + "sorted", make_slice(ops_sorted));
    }

    void write_file() {
        { write_init_file(this->init_file); }
        {
            auto test_ops =
                generate_tasks(this->pos, test_n, false, 0.0, this->bias);
            write_ops_to_file(test_file, make_slice(test_ops));
        }
    }

    void generate_microbenchmark_one_type(int t) {
        if (t == operation_t::update_t) return;
        auto pos = sequence<double>(OPERATION_NR_ITEMS, 0);
        pos[t] = 1.0;
        for (double alpha = 0.0; alpha <= 1.2; alpha += 0.2) {
            double aalpha = alpha;
            if(alpha == 1.0) aalpha = 0.99;
            parlay::sequence<operation> ops;
            if (t == operation_t::scan_t) {
                int expected_length = 100;
                ops = generate_tasks(pos, test_n / expected_length, true, aalpha, 1.0);
            } else {
                ops = generate_tasks(pos, test_n, true, aalpha, 1.0);
            }
            char filename[500];
            sprintf(filename, "test_%d_%.1f_%d.binary", test_n, alpha, t);
            write_ops_to_file(string(filename), make_slice(ops));
            cout << filename << endl;
        }
    }

    void generate_microbenchmarks() {
        for (int t = 1; t < OPERATION_NR_ITEMS; t++) {
            generate_microbenchmark_one_type(t);
        }
    }

    void generate_YCSB_benchmark() {
        auto typechecker = [&](auto ops) {
            int x[OPERATION_NR_ITEMS];
            memset(x, 0, sizeof(x));
            for (int i = 0; i < ops.size(); i ++) {
                operation_t k = ops[i].type;
                x[k] ++;
            }
            for (int i = 0; i < OPERATION_NR_ITEMS; i ++) {
                printf("%d %d\n", i, x[i]);
            }
        };

        // YCSB A = 50% predecessor : 50% insert
        for (double alpha = 0.0; alpha < 1.1; alpha += 1.0) {
            auto pos = sequence<double>(OPERATION_NR_ITEMS, 0);
            pos[operation_t::predecessor_t] = 1.0;
            auto ops = generate_tasks(pos, test_n, true, alpha, 1.0);
            ASSERT(ops.size() == 1e8);
            for (int i = 0; i < ops.size(); i ++) {
                operation original_task = ops[i];
                if (rand() % 2 == 0) {
                    operation switched_task;
                    switched_task.type = operation_t::insert_t;
                    switched_task.tsk.i = (insert_operation){.key = original_task.tsk.p.key};
                    ops[i] = switched_task;
                }
            }
            char filename[500];
            sprintf(filename, "test_%d_%.1f_ycsb_a.binary", test_n, alpha);
            typechecker(ops);
            write_ops_to_file(string(filename), make_slice(ops));
            cout << filename << endl;
        }

        // YCSB B = 95% predecessor : 5% insert
        for (double alpha = 0.0; alpha < 1.1; alpha += 1.0) {
            auto pos = sequence<double>(OPERATION_NR_ITEMS, 0);
            pos[operation_t::predecessor_t] = 1.0;
            auto ops = generate_tasks(pos, test_n, true, alpha, 1.0);
            ASSERT(ops.size() == 1e8);
            for (int i = 0; i < ops.size(); i ++) {
                operation original_task = ops[i];
                if (rand() % 20 == 0) {
                    operation switched_task;
                    switched_task.type = operation_t::insert_t;
                    switched_task.tsk.i = (insert_operation){.key = original_task.tsk.p.key};
                    ops[i] = switched_task;
                }
            }
            char filename[500];
            sprintf(filename, "test_%d_%.1f_ycsb_b.binary", test_n, alpha);
            typechecker(ops);
            write_ops_to_file(string(filename), make_slice(ops));
            cout << filename << endl;
        }

        // YCSB C = micro predecessor
        for (double alpha = 0.0; alpha < 1.1; alpha += 1.0) {
            char filename[500];
            sprintf(filename, "test_%d_%.1f_ycsb_c.binary", test_n, alpha);
            string targetname(filename);
            sprintf(filename, "test_%d_%.1f_%d.binary", test_n, alpha, predecessor_t);
            string sourcename(filename);
            {
                struct stat statbuf;
                assert(stat(sourcename.c_str(), &statbuf) == 0);
                // otherwise source file doesn't exist
            }
            string cmd = "cp '" + sourcename + "' '" + targetname + "'";
            cout<<targetname<<endl;
            system(cmd.c_str());
        }

        // YCSB D = micro insert
        for (double alpha = 0.0; alpha < 1.1; alpha += 1.0) {
            char filename[500];
            sprintf(filename, "test_%d_%.1f_ycsb_d.binary", test_n, alpha);
            string targetname(filename);
            sprintf(filename, "test_%d_%.1f_%d.binary", test_n, alpha, insert_t);
            string sourcename(filename);
            {
                struct stat statbuf;
                assert(stat(sourcename.c_str(), &statbuf) == 0);
                // otherwise source file doesn't exist
            }
            string cmd = "cp '" + sourcename + "' '" + targetname + "'";
            cout<<targetname<<endl;
            system(cmd.c_str());
        }

        // YCSB E = 95% scan : 5% insert
        for (double alpha = 0.0; alpha < 1.1; alpha += 1.0) {
            auto pos = sequence<double>(OPERATION_NR_ITEMS, 0);
            pos[operation_t::scan_t] = 1.0;
            auto ops = generate_tasks(pos, test_n / 100, true, alpha, 1.0);
            ASSERT(ops.size() == 1e8);
            for (int i = 0; i < ops.size(); i ++) {
                operation original_task = ops[i];
                if (rand() % 20 == 0) {
                    operation switched_task;
                    switched_task.type = operation_t::insert_t;
                    switched_task.tsk.i = (insert_operation){.key = original_task.tsk.s.lkey};
                    ops[i] = switched_task;
                }
            }
            char filename[500];
            sprintf(filename, "test_%d_%.1f_ycsb_e.binary", test_n, alpha);
            typechecker(ops);
            write_ops_to_file(string(filename), make_slice(ops));
            cout << filename << endl;
        }
    }

    void generate_all_test() {
        fs::path workload_directory = "/scratch/pim_workloads";
        fs::create_directories(workload_directory);
        if (this->init_file.length() == 0) {
            write_init_file((workload_directory/"init.in").c_str());
            this->init_file = (workload_directory/"init.in");
        } else {
            fs::path init_file_path(this->init_file);
            workload_directory = init_file_path.remove_filename();
            auto ops = read_op_file(this->init_file, [&](const operation& op) {
                assert(op.type == insert_t);
            });
            init_oracle(make_slice(ops));
            printf("load finished\n");
            string sorted_input_file = this->init_file + "sorted";
            if (!filesystem::exists(sorted_input_file)) {
                auto ops_sorted =
                    parlay::sort(ops, [](const operation& a, const operation& b) {
                        return a.tsk.i.key < b.tsk.i.key;
                    });
                write_ops_to_file(sorted_input_file, make_slice(ops_sorted));
                printf("generate sorted init\n");
            }
        }

        cout << "finish init" << endl;
        filesystem::current_path(workload_directory);
        // generate_microbenchmark_one_type(operation_t::scan_t);
        // exit(0);

        generate_microbenchmarks();
        generate_YCSB_benchmark();
    }
};

class driver {
   public:
    static argparse::ArgumentParser parser() {
        argparse::ArgumentParser program;
        program.add_argument("--file", "-f")
            .help("--file [init_file] [test_file]")
            .default_value(vector<string>())
            .nargs(2);
        program.add_argument("--nocheck", "-c")
            .help(
                "stop checking the correctness of the tested data "
                "structure")
            .default_value(false)
            .implicit_value(true);
        program.add_argument("--noprint", "-t")
            .help("don't print timer name when timing")
            .default_value(false)
            .implicit_value(true);
        program.add_argument("--nodetail", "-d")
            .help("don't show detail")
            .default_value(false)
            .implicit_value(true);
        program.add_argument("--get", "-g")
            .help("-g [get_ratio]")
            .default_value(0.0)
            .scan<'g', double>();
        program.add_argument("--update", "-u")
            .help("-u [update_ratio]")
            .default_value(0.0)
            .scan<'g', double>();
        program.add_argument("--predecessor", "-p")
            .help("-p [predecessor_ratio]")
            .default_value(1.0)
            .scan<'g', double>();
        program.add_argument("--scan", "-s")
            .help("-s [scan_ratio]")
            .default_value(0.0)
            .scan<'g', double>();
        program.add_argument("--insert", "-i")
            .help("-i [insert_ratio]")
            .default_value(0.0)
            .scan<'g', double>();
        program.add_argument("--remove", "-r")
            .help("-r [remove_ratio]")
            .default_value(0.0)
            .scan<'g', double>();
        program.add_argument("--length", "-l")
            .help("-l [init_length] [test_length]")
            .nargs(2)
            .default_value(vector<int>{40000000, 20000000})
            .scan<'i', int>();
        program.add_argument("--init_batch_size")
            .help("--init_batch_size [init batch size]")
            .default_value(1000000)
            .scan<'i', int>();
        program.add_argument("--test_batch_size")
            .help("--test_batch_size [test batch size]")
            .default_value(1000000)
            .scan<'i', int>();
        program.add_argument("--output_batch_size")
            .help("--output_batch_size [batch size for output file]")
            .default_value(1000000)
            .scan<'i', int>();
        program.add_argument("--bias")
            .help("--bias [?x]")
            .default_value(1)
            .scan<'i', int>();
        program.add_argument("--top_level_threads")
            .help("--top_level_threads [#threads]")
            .default_value(1)
            .scan<'i', int>();
        program.add_argument("--wait_microsecond")
            .help("--wait_microsecond [#microsecond between each]")
            .default_value(0)
            .scan<'i', int>();
        program.add_argument("--alpha")
            .help("--alpha [?x]")
            .default_value(0.0)
            .scan<'g', double>();
        program.add_argument("--output", "-o")
            .help("--output [init_file] [test_file]")
            .default_value(vector<string>())
            .nargs(2);
        program.add_argument("--push_pull_limit_dynamic")
            .help("--push_pull_limit_dynamic [limit] (limit for pull/push, limit * 2 for shadow push)")
            .default_value(L2_SIZE)
            .scan<'i', int>();
        program.add_argument("--generate_all_test_cases")
            .help("generate all test cases [initfile]")
            .default_value(string(""));
        program.add_argument("--init_state")
            .help("init state")
            .default_value(false)
            .implicit_value(true);

        return program;
    }

    static void init() {
        skew_generator::rn_gen::init();
        init_io_managers();
    }

    static void run(frontend& f, int init_batch_size, int test_batch_size) {
        pim_skip_list_drivers = new pim_skip_list[core::num_top_level_threads];
        pim_skip_list_drivers[0].init();
        
        {
            auto init_ops = f.init_tasks();
            cpu_coverage_timer->reset();
            pim_coverage_timer->reset();
            core::execute(make_slice(init_ops), init_batch_size,
                          init_batch_size, 1);
        }
        total_communication = 0;
        total_actual_communication = 0;

        for (int i = 0; i < core::num_top_level_threads; i ++) {
            pim_skip_list_drivers[i].push_pull_limit_dynamic = core::push_pull_limit_dynamic;
        }

        dpu_energy_stats(false);
        reset_all_timers();
        {
            auto test_ops = f.test_tasks();
            cpu_coverage_timer->reset();
            pim_coverage_timer->reset();

#ifdef USE_PAPI
            papi_init_program(parlay::num_workers());
            papi_reset_counters();
            papi_turn_counters(true);
            papi_check_counters(parlay::worker_id());
            papi_wait_counters(true, parlay::num_workers());
#endif

            core::execute(make_slice(test_ops), test_batch_size,
                          test_batch_size, core::num_top_level_threads);

#ifdef USE_PAPI
            papi_turn_counters(false);
            papi_check_counters(parlay::worker_id());
            papi_wait_counters(false, parlay::num_workers());
#endif

            print_all_timers(print_type::pt_full);
            print_all_timers(print_type::pt_name);
            print_all_timers(print_type::pt_time);
            print_all_timers_average();
            cpu_coverage_timer->print(pt_full);
            pim_coverage_timer->print(pt_full);

#ifdef USE_PAPI
            papi_print_counters(1);
#endif
        }

        dpu_energy_stats(false);
        delete[] pim_skip_list_drivers;
    }

    static void exec(int argc, char* argv[]) {
        auto program = parser();

        try {
            program.parse_args(argc, argv);
        } catch (const std::runtime_error& err) {
            std::cerr << err.what() << std::endl;
            std::cerr << program;
            std::exit(1);
        }

        parlay::sequence<double> posibility_of_tasks(OPERATION_NR_ITEMS, 0.0);
        posibility_of_tasks[1] = program.get<double>("-g");
        posibility_of_tasks[2] = program.get<double>("-u");
        posibility_of_tasks[3] = program.get<double>("-p");
        posibility_of_tasks[4] = program.get<double>("-s");
        posibility_of_tasks[5] = program.get<double>("-i");
        posibility_of_tasks[6] = program.get<double>("-r");

        core::check_result = (program["--nocheck"] == false);
        timer::print_when_time = (program["--noprint"] == false);
        timer::default_detail = (program["--nodetail"] == false);
        int bias = program.get<int>("--bias");
        core::push_pull_limit_dynamic = program.get<int>("--push_pull_limit_dynamic");

        core::num_top_level_threads = program.get<int>("--top_level_threads");

        ASSERT(core::num_top_level_threads >= 1);
        cout << "thread: " << core::num_top_level_threads << endl;

        double alpha = program.get<double>("--alpha");
        auto files = program.get<vector<string>>("-f");
        auto output_file = program.get<vector<string>>("-o");
        auto ns = program.get<vector<int>>("-l");
        assert(ns.size() == 2);
        int init_n = ns[0];
        int test_n = ns[1];

        int init_batch_size = program.get<int>("--init_batch_size");
        int test_batch_size = program.get<int>("--test_batch_size");
        int output_batch_size = program.get<int>("--output_batch_size");

        if (program.is_used("--generate_all_test_cases") == true) {
            cout << "start generating all tests" << endl;
            string init_file = program.get<string>("--generate_all_test_cases");
            frontend_testgen frontend(init_n, test_n, posibility_of_tasks, bias,
                                      init_file, "", output_batch_size);
            frontend.generate_all_test();
        } else if (files.size() > 0) {  // test from file
            assert(files.size() == 2);
            printf("Test from file: [%s] [%s]\n", files[0].c_str(),
                   files[1].c_str());
            int in = -1, tn = -1;
            if (program.is_used("-l")) {
                in = ns[0];
                tn = ns[1];
            }
            frontend_by_file frontend(files[0], files[1], in, tn);
            run(frontend, init_batch_size, test_batch_size);
        } else if (output_file.size() > 0) {  // print test file
            assert(output_file.size() == 2);
            printf("To generated file:\n");
            for (int i = 1; i < OPERATION_NR_ITEMS; i++) {
                printf("posibility_of_tasks[%d]=%lf\n", i, posibility_of_tasks[i]);
            }
            printf("\n");

            frontend_testgen frontend(init_n, test_n, posibility_of_tasks, bias,
                                      output_file[0], output_file[1],
                                      output_batch_size);
            frontend.write_file();
        } else {  // in memory test
            printf("Test with generated data:\n");
            for (int i = 1; i < OPERATION_NR_ITEMS; i++) {
                printf("posibility_of_tasks[%d]=%lf\n", i, posibility_of_tasks[i]);
            }
            printf("\n");

            auto ns = program.get<vector<int>>("-l");
            assert(ns.size() == 2);
            int init_n = ns[0];
            int test_n = ns[1];
            frontend_by_generation frontend(init_n, test_n, posibility_of_tasks, bias,
                                            init_batch_size, test_batch_size);
            run(frontend, init_batch_size, test_batch_size);
        }

        // print_all_timers(print_type::pt_full);
        // print_all_timers(print_type::pt_time);
        // print_all_timers(print_type::pt_name);
        cout << "total communication" << total_communication.load() << endl;
        cout << "total actual communication"
             << total_actual_communication.load() << endl;
    }
};
