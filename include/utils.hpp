#ifndef UTILS_H_
#define UTILS_H_
#include <iostream>
#include <math.h>
#include <string>
#include <vector>
#include <utility>
#include <fstream>
#include <thread>
#include <regex>
#include <map>
#include <algorithm>
#include <dlpack/dlpack.h>
#include <sys/time.h>
#include <stdlib.h>
#include <emmintrin.h>
#include <immintrin.h>
#include <unistd.h>
#include <sched.h>
#include <pthread.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
#include <emmintrin.h>
#include <immintrin.h>
#define NEW_TASK_PROFILING 1
#define LAYER_PARAM_LEN_MAX 256
#define TENSOR_DIMS 4
#define DATA_BITS 32
#define DATA_LANE 1
#define DEVICE_ID 0 // ?
#define DATA_TYPE kDLFloat
#define DEVICE_TYPE kDLCPU
#define WORKER_NUM 8
#define MULTI_VERSION_SAMPLING_INTEVAL 50
#define FLUSH_L3_CACHE 0
#define MULTI_VERSION 1
#define FINE_GRAIN 1
#define CORE_NUM_STRATEGY 0 // 0-Max, 1-Min, 2-alpha-MAX.
#define MIN_CPU_UNIT 8
#define ALPHA 1
#define CORE_NUM_DECAY 0
#define BETA 0.9           // Advanced Finished Ratio.
#define SUB_MODEL_SIZE 20
#define TOTAL_CPU_UNIT (sysconf(_SC_NPROCESSORS_ONLN))
#define CPU_STRONG_AFFINITY_BINDING 0
#define TRY_PREEMPTING 0
#define LIB_PATH "/mnt/e/MultiVersion/auto_scheduler/5libs/"

enum layer_type {
    conv1d,
    conv2d,
    conv3d,
    gemm,
    pooling,
    batchnorm,
    relu,
    len_layer_type,
};

enum task_type {
    classification_light_mobilenet = 1,
    classification_light_efficientnet = 2,
    classification_heavy_resnet50 = 3,
    classification_heavy_googlenet = 4,
    detection_light_tinyyolov2 = 5,
    detection_heavy_ssd = 6,
    nmt_heavy_bert = 7,
    num_task_type = 8,
};

float elapsed(struct timeval a, struct timeval b) {
    return 1000000.0 * (b.tv_sec - a.tv_sec) + 1.0 * (b.tv_usec - a.tv_usec);
}

float get_timestamp(struct timeval st) {
    return 1000000.0 * st.tv_sec + 1.0 * st.tv_usec;
}

template <typename T>
float max(std::vector<T> et) {
    sort(et.begin(), et.end(), [](T a, T b){return a > b;});
    return et[0];
}

template <typename T>
float min(std::vector<T> et) {
    sort(et.begin(), et.end(), [](T a, T b){return a < b;});
    return et[0];
}

template <typename T>
float median(std::vector<T> et) {
    sort(et.begin(), et.end(), [](T a, T b){return a > b;});
    if (et.size() % 2 == 0) {
	    return (et[et.size() / 2 - 1] + et[et.size() / 2]) / 2.0f;
    }
    else {
	    return et[(et.size() - 1) / 2];
    }
}

template <typename T>
float mean(std::vector<T> et) {
    float sum = 0.0f;
    for (auto&& i : et) {
        sum += i;
    }
    return sum / (1.0 * et.size());
}

void find_min_idx(int* l, int len, int* target_worker, int* target_worker_len, int* current_launched) {
    int idx = 0;
    int min_res = 114514;
    int sum = 0;
    for (int i = 0; i < len; i++) {
        if (*(l + i) < min_res) {
            min_res = *(l + i);
            idx = i;
            sum += *(l + i);
        }
    }
    *target_worker = idx;
    *target_worker_len = min_res;
    *current_launched = sum;
}

std::map <std::string, int> task_mapping;

#endif