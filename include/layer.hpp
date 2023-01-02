#ifndef LAYER_H_
#define LAYER_H_

#include "utils.hpp"
#include <string>

class layer {
private:
    size_t layer_id;
    std::string belonging;
    layer_type lt;
    std::string lib_path;
    std::string meta_param_string;
    float op_count;// MACs
    size_t sel_version;
    size_t core_num;
    size_t actual_core_num;
    cpu_set_t layer_affinity;
    tvm::runtime::PackedFunc f;
    std::vector<tvm::runtime::PackedFunc> funcs;
    std::vector <DLTensor*> layer_io;
    float qos;
    float qos_accum;
    float pre_layers_executed_time;
    bool execute_at_sub_optimal;
    int HI, WI, CI, CO, HO, WO, KH, KW, SH, SW, PH, PW, GP, M, N, K;
public:
    layer() {
        op_count = 0.0f;
        core_num = 1;
        sel_version = 0;
        qos = 0.0f;
        qos_accum = 0.0f;
        pre_layers_executed_time = 0.0f;
        layer_io.clear();
        CPU_ZERO(&layer_affinity);
        for (int i = 0; i < 64; i++) CPU_SET(i, &layer_affinity);
        execute_at_sub_optimal = false;
    }

    // DONT USE
    // layer(const layer_type& __lt, const std::string& __meta_param_string);

    void layerExtractFunction() {
        std::string f_name = "";
        switch (lt) {
            case gemm: {
                f_name += "gemm_";
                std::regex filter("-");
                std::vector <std::string> gemm_params(std::sregex_token_iterator(meta_param_string.begin(), meta_param_string.end(), filter, -1), std::sregex_token_iterator());
                M = stoi(gemm_params[0]);
                N = stoi(gemm_params[1]);
                K = stoi(gemm_params[2]);
                f_name += std::to_string(M) + "_" + std::to_string(N) + "_" + std::to_string(K);
                std::string _f_name;
                tvm::runtime::Module _dyn_mod;
                tvm::runtime::PackedFunc _f;
                for (int v = 0; v < 5; v++) {
                    std::string _f_name = f_name + ("_" + std::to_string(v));
                    _dyn_mod = tvm::runtime::Module::LoadFromFile(lib_path + _f_name + ".so");
                    _f = _dyn_mod.GetFunction(_f_name);
                    funcs.push_back(_f);
                }

                DLTensor *A;
                DLTensor *B;
                DLTensor *C;
                int64_t shape_A[2] = {M, K};
                int64_t shape_B[2] = {N, K};
                int64_t shape_C[2] = {M, N};
                TVMArrayAlloc(shape_A, 2, DATA_TYPE, DATA_BITS, DATA_LANE, DEVICE_TYPE, DEVICE_ID, &A);
                TVMArrayAlloc(shape_B, 2, DATA_TYPE, DATA_BITS, DATA_LANE, DEVICE_TYPE, DEVICE_ID, &B);
                TVMArrayAlloc(shape_C, 2, DATA_TYPE, DATA_BITS, DATA_LANE, DEVICE_TYPE, DEVICE_ID, &C);
                layer_io.push_back(A);
                layer_io.push_back(B);
                layer_io.push_back(C);
                break;
            }
            case conv2d: {

                // Extract the function from library
                f_name += "conv2d_";
                std::regex filter("-");
                std::vector <std::string> conv_params(std::sregex_token_iterator(meta_param_string.begin(), meta_param_string.end(), filter, -1), std::sregex_token_iterator());
                HI = stoi(conv_params[0]);
                WI = stoi(conv_params[1]);
                CI = stoi(conv_params[2]);
                CO = stoi(conv_params[3]);
                KH = stoi(conv_params[4]);
                KW = stoi(conv_params[5]);
                SH = stoi(conv_params[8]);
                SW = stoi(conv_params[9]);
                PH = stoi(conv_params[6]);
                PW = stoi(conv_params[7]);
                GP = stoi(conv_params[10]);
                // int HO = stoi(conv_params[10]);
                // int WO = stoi(conv_params[11]);
                HO = (int) ((HI + 2 * PH - KH + 1) / SH);
                WO = (int) ((WI + 2 * PW - KW + 1) / SW);
                op_count = 1.0 * CO * CI * KH * KW * HO * WO;
                f_name += std::to_string(HI) + "_" + std::to_string(WI) + "_" + std::to_string(CI) + "_" + std::to_string(CO) + "_" + std::to_string(KH) + "_" + std::to_string(KW) + "_" + std::to_string(PH) + "_" + std::to_string(PW) + "_" + std::to_string(SH) + "_" + std::to_string(SW) + "_" + std::to_string(GP);
                std::string _f_name;
                if (sel_version != -1) {
                    _f_name = f_name + ("_" + std::to_string(sel_version));
                }
                tvm::runtime::Module dyn_mod = tvm::runtime::Module::LoadFromFile(lib_path + _f_name + ".so");
                f = dyn_mod.GetFunction(_f_name);

                tvm::runtime::Module _dyn_mod;
                tvm::runtime::PackedFunc _f;
                for (int v = 0; v < 5; v++) {
                    std::string _f_name = f_name + ("_" + std::to_string(v));
                    _dyn_mod = tvm::runtime::Module::LoadFromFile(lib_path + _f_name + ".so");
                    _f = _dyn_mod.GetFunction(_f_name);
                    funcs.push_back(_f);
                }

                // Prepare the tensor for the function
                DLTensor* data;
                DLTensor* weight;
                DLTensor* bias;
                DLTensor* output;
                int64_t shape_data[4] = {1, CI, HI, WI};
                int64_t shape_weight[4] = {CO, CI, KH, KW};
                int64_t shape_bias[4] = {1, CO, 1, 1};
                int64_t shape_output[4] = {1, CO, HO, WO};
                TVMArrayAlloc(shape_data, TENSOR_DIMS, DATA_TYPE, DATA_BITS, DATA_LANE, DEVICE_TYPE, DEVICE_ID, &data);
                TVMArrayAlloc(shape_weight, TENSOR_DIMS, DATA_TYPE, DATA_BITS, DATA_LANE, DEVICE_TYPE, DEVICE_ID, &weight);
                TVMArrayAlloc(shape_bias, TENSOR_DIMS, DATA_TYPE, DATA_BITS, DATA_LANE, DEVICE_TYPE, DEVICE_ID, &bias);
                TVMArrayAlloc(shape_output, TENSOR_DIMS, DATA_TYPE, DATA_BITS, DATA_LANE, DEVICE_TYPE, DEVICE_ID, &output);
                layer_io.push_back(data);
                layer_io.push_back(weight);    
                layer_io.push_back(bias);
                layer_io.push_back(output);               
                break;
            }
            default: {
                std::cout << "Unknown OP" << std::endl;
            }
        }
    }

    void layerExecuteAsBackground(int ver_idx=4) {
        f = funcs[ver_idx];
        core_num = 32;
        //CPU_ZERO(&layer_affinity);
        //for (int i = 4; i < 8; i++) CPU_SET(i, &layer_affinity);
        //sched_setaffinity(0, sizeof(cpu_set_t), &layer_affinity);
        while (true) {
#if FLUSH_L3_CACHE == 1
            auto begin = layer_io[0]->data;
            auto cache_line_size = sysconf(_SC_LEVEL1_DCACHE_LINESIZE); 
            for (uintptr_t uptr = (uintptr_t)begin & ~(cache_line_size - 1); uptr < (uintptr_t)begin + CI * HI * WI; uptr += cache_line_size) {
                _mm_clflush(reinterpret_cast<const void*>(uptr));
            }
            begin = layer_io[1]->data;
            for (uintptr_t uptr = (uintptr_t)begin & ~(cache_line_size - 1); uptr < (uintptr_t)begin + CO * CI * KH * KW; uptr += cache_line_size) {
                _mm_clflush(reinterpret_cast<const void*>(uptr));
            }
            begin = layer_io[2]->data;
            for (uintptr_t uptr = (uintptr_t)begin & ~(cache_line_size - 1); uptr < (uintptr_t)begin + CO; uptr += cache_line_size) {
                _mm_clflush(reinterpret_cast<const void*>(uptr));
            }
            begin = layer_io[3]->data;
            for (uintptr_t uptr = (uintptr_t)begin & ~(cache_line_size - 1); uptr < (uintptr_t)begin + CO * HO * WO; uptr += cache_line_size) {
                _mm_clflush(reinterpret_cast<const void*>(uptr));
            }
#endif
            if (lt == conv2d) f(layer_io[0], layer_io[1], layer_io[2], layer_io[3]);
            else if (lt == gemm) f(layer_io[0], layer_io[1], layer_io[2]);
        }
    }

    void layerExecute(float* eps_time, int rank) {
/*
        for (int v = 0; v < 5; v++) {
            tvm::runtime::PackedFunc _f = funcs[v];
            for (int u = 0; u < 10; u++) {
                _f(layer_io[0], layer_io[1], layer_io[2], layer_io[3]);
            }
        }
*/
        setenv("TVM_NUM_THREADS", std::to_string(core_num).c_str(), 1);
        setenv("TVM_BIND_THREADS", "0", 1);
        struct timeval st, ed;
        std::vector <float> time;
        time.clear();
        // sched_setaffinity(0, sizeof(cpu_set_t), &(layer_affinity));
/*
        gettimeofday(&st, NULL);
        std::thread t([this] {layerRun();});
        pthread_setaffinity_np(t.native_handle(), sizeof(cpu_set_t), &(layer_affinity));
        t.join();
        gettimeofday(&ed, NULL);
        *eps_time = elapsed(st, ed) / 100.0;
        // std::cout << belonging << " - " << layer_id << " : " << elapsed(st, ed) / 10.0 << std::endl;
*/
#if MULTI_VERSION == 0
/*
        int idx = 0, mintime = 1e10;
        struct timeval _st, _ed;
        for (int v = 0; v < 5; v++) {
            f = funcs[v];
            for (int _i = 0; _i < 50; _i++) f(layer_io[0], layer_io[1], layer_io[2], layer_io[3]);
            gettimeofday(&_st, NULL);
            for (int _i = 0; _i < 50; _i++) f(layer_io[0], layer_io[1], layer_io[2], layer_io[3]);
            gettimeofday(&_ed, NULL);
            float _eps = elapsed(_st, _ed);
            if (_eps < mintime) {
                idx = v;
                mintime = _eps;
            }
        }
        f = funcs[4];
        for (int i = 0; i < 100; i++) {
#if FLUSH_L3_CACHE == 1
            auto begin = layer_io[0]->data;
            auto cache_line_size = sysconf(_SC_LEVEL1_DCACHE_LINESIZE); 
            for (uintptr_t uptr = (uintptr_t)begin & ~(cache_line_size - 1); uptr < (uintptr_t)begin + CI * HI * WI; uptr += cache_line_size) {
                _mm_clflush(reinterpret_cast<const void*>(uptr));
            }
            begin = layer_io[1]->data;
            for (uintptr_t uptr = (uintptr_t)begin & ~(cache_line_size - 1); uptr < (uintptr_t)begin + CO * CI * KH * KW; uptr += cache_line_size) {
                _mm_clflush(reinterpret_cast<const void*>(uptr));
            }
            begin = layer_io[2]->data;
            for (uintptr_t uptr = (uintptr_t)begin & ~(cache_line_size - 1); uptr < (uintptr_t)begin + CO; uptr += cache_line_size) {
                _mm_clflush(reinterpret_cast<const void*>(uptr));
            }
            begin = layer_io[3]->data;
            for (uintptr_t uptr = (uintptr_t)begin & ~(cache_line_size - 1); uptr < (uintptr_t)begin + CO * HO * WO; uptr += cache_line_size) {
                _mm_clflush(reinterpret_cast<const void*>(uptr));
            }
#endif
            gettimeofday(&st, NULL);
            f(layer_io[0], layer_io[1], layer_io[2], layer_io[3]);
            gettimeofday(&ed, NULL);
            if (i > 50) {
                float eps = elapsed(st, ed);
                time.push_back(eps);
            }
        }
        // std::cout << median(time) << std::endl;
        // std::cout << meta_param_string << " - " << idx << std::endl;
        float med_eps = median(time);
*/
        f = funcs[2];
        std::ifstream fin;
        float overhead = 0.0f;
        struct timeval overhead_st, overhead_ed;
        int avail_slot = 64, tmp;
        sched_setaffinity(0, sizeof(cpu_set_t), &layer_affinity);
        for (int i = 0; i < 100; i++) {
#if TRY_PREEMPTING == 1
            gettimeofday(&overhead_st, NULL);
            if (i % 25 == 24) {
                avail_slot = 64;
                for (int w = 1; w < WORKER_NUM; w++) {
                    fin.open("Util_" + std::to_string(w), std::ios::in);
                    fin >> tmp;
                    fin.close();
                    avail_slot -= tmp;
                }
                if (avail_slot > actual_core_num) {
                    setenv("TVM_NUM_THREADS", std::to_string(avail_slot).c_str(), 1);
                }
            }
            gettimeofday(&overhead_ed, NULL);
#endif
#if FLUSH_L3_CACHE == 1
            auto begin = layer_io[0]->data;
            auto cache_line_size = sysconf(_SC_LEVEL1_DCACHE_LINESIZE); 
            for (uintptr_t uptr = (uintptr_t)begin & ~(cache_line_size - 1); uptr < (uintptr_t)begin + CI * HI * WI; uptr += cache_line_size) {
                _mm_clflush(reinterpret_cast<const void*>(uptr));
            }
            begin = layer_io[1]->data;
            for (uintptr_t uptr = (uintptr_t)begin & ~(cache_line_size - 1); uptr < (uintptr_t)begin + CO * CI * KH * KW; uptr += cache_line_size) {
                _mm_clflush(reinterpret_cast<const void*>(uptr));
            }
            begin = layer_io[2]->data;
            for (uintptr_t uptr = (uintptr_t)begin & ~(cache_line_size - 1); uptr < (uintptr_t)begin + CO; uptr += cache_line_size) {
                _mm_clflush(reinterpret_cast<const void*>(uptr));
            }
            begin = layer_io[3]->data;
            for (uintptr_t uptr = (uintptr_t)begin & ~(cache_line_size - 1); uptr < (uintptr_t)begin + CO * HO * WO; uptr += cache_line_size) {
                _mm_clflush(reinterpret_cast<const void*>(uptr));
            }
#endif
            gettimeofday(&st, NULL);            
            if (lt == conv2d) f(layer_io[0], layer_io[1], layer_io[2], layer_io[3]);
            else if (lt == gemm) f(layer_io[0], layer_io[1], layer_io[2]);
            gettimeofday(&ed, NULL);
            if (i > 50) {
                float eps = elapsed(st, ed);
                time.push_back(eps);
            }
#if TRY_PREEMPTING == 1
            overhead += elapsed(overhead_st, overhead_ed);
#else
            overhead = 0.0;
#endif
        }
        float med_eps = median(time);
        // std::cout << op_count / med_eps << std::endl;
        *eps_time = med_eps;
#else
        int idx = 0, mintime = 0;
        tvm::runtime::PackedFunc _f;
        struct timeval _st, _ed;
        std::vector <int> ver_sel;
        ver_sel.clear();
        for (int pf_time = 0; pf_time < 2; pf_time++) {
            for (int v = 0; v < 5; v++) {
                _f = funcs[v];
                gettimeofday(&_st, NULL);
                for (int _i = 0; _i < 5; _i++) {
                    
                    if (lt == conv2d) _f(layer_io[0], layer_io[1], layer_io[2], layer_io[3]);
                    else if  (lt == gemm) _f(layer_io[0], layer_io[1], layer_io[2]);
                }
                gettimeofday(&_ed, NULL);
                float _eps = elapsed(_st, _ed);
                if (_eps > mintime) {
                    idx = v;
                    mintime = _eps;
                }
            }
            _f = funcs[idx];
            ver_sel.push_back(idx);
            for (int i = 0; i < 100; i++) {
                gettimeofday(&st, NULL);
#if FLUSH_L3_CACHE == 1
                auto begin = layer_io[0]->data;
                auto cache_line_size = sysconf(_SC_LEVEL1_DCACHE_LINESIZE); 
                for (uintptr_t uptr = (uintptr_t)begin & ~(cache_line_size - 1); uptr < (uintptr_t)begin + CI * HI * WI; uptr += cache_line_size) {
                    _mm_clflush(reinterpret_cast<const void*>(uptr));
                }
                begin = layer_io[1]->data;
                for (uintptr_t uptr = (uintptr_t)begin & ~(cache_line_size - 1); uptr < (uintptr_t)begin + CO * CI * KH * KW; uptr += cache_line_size) {
                    _mm_clflush(reinterpret_cast<const void*>(uptr));
                }
                begin = layer_io[2]->data;
                for (uintptr_t uptr = (uintptr_t)begin & ~(cache_line_size - 1); uptr < (uintptr_t)begin + CO; uptr += cache_line_size) {
                    _mm_clflush(reinterpret_cast<const void*>(uptr));
                }
                begin = layer_io[3]->data;
                for (uintptr_t uptr = (uintptr_t)begin & ~(cache_line_size - 1); uptr < (uintptr_t)begin + CO * HO * WO; uptr += cache_line_size) {
                    _mm_clflush(reinterpret_cast<const void*>(uptr));
                }
#endif
                if (lt == conv2d) _f(layer_io[0], layer_io[1], layer_io[2], layer_io[3]);
                else if  (lt == gemm) _f(layer_io[0], layer_io[1], layer_io[2]);
                gettimeofday(&ed, NULL);
                if (i > 50) {
                    float eps = elapsed(st, ed);
                    time.push_back(eps);
                }
            }
        }
        // std::cout << median(time) << std::endl;
        // std::cout << meta_param_string << " - ";
        // for (auto&& vs : ver_sel) std::cout << vs << " ";
        // std::cout << std::endl;
        float med_eps = median(time);
        *eps_time = med_eps;
#endif
    }

/*
    void layerRun(int iter=100) {
        // f(layer_io[0], layer_io[1], layer_io[2], layer_io[3]);
#if MULTI_VERSION == 0
        for (int i = 0; i < iter; i++) {
            f(layer_io[0], layer_io[1], layer_io[2], layer_io[3]);
        }
#else
        int idx = 0, mintime = 1e10;
        tvm::runtime::PackedFunc _f;
        struct timeval _st, _ed;
        for (int pf_time = 0; pf_time < 2; pf_time++) {
            for (int v = 0; v < 5; v++) {
                _f = funcs[v];
                gettimeofday(&_st, NULL);
                for (int _i = 0; _i < 1; _i++) _f(layer_io[0], layer_io[1], layer_io[2], layer_io[3]);
                gettimeofday(&_ed, NULL);
                float _eps = elapsed(_st, _ed);
                if (_eps < mintime) {
                    idx = v;
                    mintime = _eps;
                }
            }
            _f = funcs[idx];
            // std::cout << "Using Funcs: " << idx << std::endl;
            for (int j = 0; j < PROF_INTEVAL; j++) {
                f(layer_io[0], layer_io[1], layer_io[2], layer_io[3]);
            }
        }
#endif
    }
    */

    friend class request;
};

#endif