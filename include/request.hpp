#ifndef REQUEST_H_
#define REQUEST_H_

#include "utils.hpp"
#include "layer.hpp"
#include "task.hpp"

class request {
private:
    std::vector <layer> layer_list;
    std::vector <std::vector<std::string>> layer_prof_time;
    float qos_requirement; // ms
    int req_id;
    float total_op_count;
public:
    explicit request();

    request(const task& __task, int mpi_rank, int whoami_in_queue, int __req_id, int __qos_requirement=15.0f) {
        req_id = __req_id;
        total_op_count = 0.0f;
        layer_prof_time = __task.layer_profiling_result;
        qos_requirement = __qos_requirement;
        for (int i = 0; i < __task.layer_num; i++) {
            layer _l;
            _l.belonging = __task.template_name + "_" + std::to_string(task_mapping[__task.template_name]) + "_" + std::to_string(mpi_rank) + "_" + std::to_string(whoami_in_queue);
            _l.layer_id = i;
            _l.lib_path = LIB_PATH;
            _l.meta_param_string = __task.layer_meta_params_string[i].second;
            if (__task.layer_meta_params_string[i].first == "CONV") {
                _l.lt = conv2d;
            }
            else if (__task.layer_meta_params_string[i].first == "RELU") {
                _l.lt = relu;
            }
            else if (__task.layer_meta_params_string[i].first == "GEMM") {
                _l.lt = gemm;
            }
            layer_list.push_back(_l);
        }
    }

    void requestCompile() {
        // std::cout << "Compiling Request ..." << std::endl;
        for (int i = 0; i < layer_list.size(); i++) {
            layer_list[i].layerExtractFunction();
            total_op_count += layer_list[i].op_count;
        }
        // std::cout << "Compiling Request Finished" << std::endl;
    }

    void requestSchedule(int emprical_core_num=-1) {
        //for (int i = 0; i < layer_list.size(); i++) layer_list[i].core_num = __core_num;
        if (NEW_TASK_PROFILING == 1) {
            for (int i = 0; i < layer_list.size(); i++) {
                layer_list[i].core_num = emprical_core_num;
            }
            return;
        }
        if (FINE_GRAIN == 0) {
            std::vector <float> time_cost;
            time_cost.clear();
            int core_num_idx = -1;
            for (int u = 0; u < (TOTAL_CPU_UNIT / MIN_CPU_UNIT); u++) {
                float sum = 0.0;
                for (int i = 0; i < layer_list.size(); i++) {
                    sum += atof(layer_prof_time[i][u].c_str()) / 1000.0;
                }
                if (sum - ALPHA * qos_requirement < 0.0f) {
                    core_num_idx = u;
                    break;
                }
            }
            int core_number = (1 + core_num_idx) * MIN_CPU_UNIT;
            // core_number = 56;
            for (int i = 0; i < layer_list.size(); i++) {
                layer_list[i].core_num = core_number;
            }
        }
        else {
            int sub_model_num = (int) (layer_list.size() / SUB_MODEL_SIZE);
            for (int i = 0; i < layer_list.size(); i+=SUB_MODEL_SIZE) {
                float sub_model_op_count = 0.0f;
                for (int j = i; j < std::min(i + SUB_MODEL_SIZE, (int)layer_list.size()); j++) {
                    sub_model_op_count += layer_list[j].op_count;
                    layer_list[j].qos = qos_requirement * layer_list[j].op_count / total_op_count;
                    if (j == 0) {
                        layer_list[j].qos_accum = layer_list[j].qos;
                    }
                    else {
                        layer_list[j].qos_accum = layer_list[j].qos + layer_list[j - 1].qos_accum;
                    }
                }
                float sub_model_qos = qos_requirement * sub_model_op_count / total_op_count;
                int sub_model_core_num_idx = -1;
                for (int u = 0; u < TOTAL_CPU_UNIT / MIN_CPU_UNIT; u++) {
                    float sub_model_time = 0.0f;
                    for (int j = i; j < std::min(i + SUB_MODEL_SIZE, (int)layer_list.size()); j++) {
                        sub_model_time += atof(layer_prof_time[j][u].c_str()) / 1000.0;
                    }
                    if (sub_model_time - ALPHA * sub_model_qos < 0.0f) {
                        sub_model_core_num_idx = u;
                        break;
                    }
                }
                if (sub_model_core_num_idx == -1) sub_model_core_num_idx = TOTAL_CPU_UNIT / MIN_CPU_UNIT - 1;
                sub_model_core_num_idx = emprical_core_num == -1 ? sub_model_core_num_idx : emprical_core_num;
                for (int j = i; j < std::min(i + SUB_MODEL_SIZE, (int)layer_list.size()); j++) {
                    layer_list[j].core_num = (1 + sub_model_core_num_idx) * MIN_CPU_UNIT;
                    // std::cout << layer_list[j].core_num << " " << layer_list[j].qos << std::endl;
                    // layer_list[j].core_num = 16;
                }
            }
            
        }
        
    }

    void requestScheduleAsBackground(int core_number) {
        for (int i = 0; i < layer_list.size(); i++) {
            layer_list[i].core_num = core_number;
        }
    }

    void requestRunAsBackground() {
        for (int i = 0; i < layer_list.size(); i++) {
            setenv("TVM_NUM_THREADS", std::to_string(layer_list[i].core_num).c_str(), 1);
            setenv("TVM_BIND_THREADS", "0", 1);
            layer_list[i].layerExecuteAsBackground();
        }
    }
/*
    void requestExecute(bool* __is_satisfied, float* __latency, int rank) {
        float eps_time, total = 0.0f, wait = 0.0f;
        bool print_eps_time = false;
        int origin_core = 0;
        struct timeval wait_st, wait_ed;
        for (int i = 0; i < layer_list.size(); i++) {
            gettimeofday(&wait_st, NULL);
            int cpu_util_cur = 0, _cuc;
SPINCHECK:
            for (int w = 1; w <= WORKER_NUM; w++) {
                if (w == rank) continue;
                std::ifstream fin("Util_" + std::to_string(w), std::ios::in);
                fin >> _cuc;
                fin.close();
                cpu_util_cur += _cuc;
            }
            if (cpu_util_cur == TOTAL_CPU_UNIT) goto SPINCHECK;
            std::ofstream fout("Util_" + std::to_string(rank), std::ios::out);
            print_eps_time = false;
            origin_core = layer_list[i].core_num;
            if (layer_list[i].core_num <= 64 - cpu_util_cur) {
                fout << layer_list[i].core_num << "\n";
            }
            else {
                //std::cout << "Conflicting @ Available: " << 64 - cpu_util_cur << ", Required: " << layer_list[i].core_num << std::endl;
                print_eps_time = true;
                fout << 64 - cpu_util_cur << "\n";
                layer_list[i].core_num = 64 - cpu_util_cur;
            }
            fout.close();
            gettimeofday(&wait_ed, NULL);
            setenv("TVM_NUM_THREADS", std::to_string(layer_list[i].core_num).c_str(), 1);
            setenv("TVM_BIND_THREADS", "0", 1);
            layer_list[i].layerExecute(&eps_time);
            std::ofstream _fout("Util_" + std::to_string(rank), std::ios::out);
            _fout << 0 << "\n";
            _fout.close();
            total += eps_time;
            if (print_eps_time == true) {
                // std::cout << eps_time << " " << origin_core << " " << layer_list[i].core_num << std::endl;
            }
            wait += elapsed(wait_st, wait_ed);
            // std::cout << eps_time << std::endl;
#if CORE_NUM_DECAY == 1
            if (layer_list[i].qos_accum - total < 0.0f) {
                for (int j = i + 1; j < std::min((int)layer_list.size(), i + 10); j++) layer_list[j].core_num = std::max((size_t) TOTAL_CPU_UNIT, layer_list[j].core_num + 8);
            }
            else if (layer_list[i].qos_accum - BETA * total > 0.0f) {
                for (int j = i + 1; j < std::min((int)layer_list.size(), i + 10); j++) layer_list[j].core_num = std::min((size_t) MIN_CPU_UNIT, layer_list[j].core_num - 8);
            }
#endif
        }
        *__latency = total;
        // std::cout << "Req: " << req_id << " Finished @ Time: " << total << std::endl;
        std::cout  << req_id << " Finished @ " << total / 1000.0 << ", Wait time =" << wait / 1000000.0 << ", QoS Requirement is: " << qos_requirement << std::endl;
        if (total / 1000.0 - 15.0 < 0.0) *__is_satisfied = true;
    }
*/
    void requestExecute(bool* __is_satisfied, float* __latency, int rank) {
        struct timeval wait_st, wait_ed;
        std::ifstream fin;
        std::ofstream fout;
        float total = 0.0f, wait = 0.0f;
        for (int i = 0; i < layer_list.size(); i+=SUB_MODEL_SIZE) {
/*
            int cur_cpu_util = 0, worker_cpu_util, actual_core;
            gettimeofday(&wait_st, NULL);
EMPTY_SLOT_SPIN_CHECK:
            for (int w = 1; w < WORKER_NUM; w++) {
                if (w == rank) continue;
                fin.open("Util_" + std::to_string(w), std::ios::in);
                fin >> worker_cpu_util;
                fin.close();
                cur_cpu_util += worker_cpu_util;
            }
            if (cur_cpu_util == TOTAL_CPU_UNIT) goto EMPTY_SLOT_SPIN_CHECK;
            gettimeofday(&wait_ed, NULL);
            actual_core = std::min(layer_list[i].core_num, (size_t) (TOTAL_CPU_UNIT - cur_cpu_util));
            if (actual_core < layer_list[i].core_num) {
                // std::cout << "PR!" << std::endl;
                layer_list[i].execute_at_sub_optimal = true;
            }
            fout.open("Util_" + std::to_string(rank), std::ios::out);
            fout << actual_core << "\n";
            fout.close();
*/
            setenv("TVM_NUM_THREADS", std::to_string(layer_list[i].core_num).c_str(), 1);
            setenv("TVM_BIND_THREADS", "0", 1);
            for (int j = i; j < std::min(i + SUB_MODEL_SIZE, (int) layer_list.size()); j++) {
                // layer_list[j].actual_core_num = actual_core;

                float eps_time = 0.0f;
                layer_list[j].layerExecute(&eps_time, rank);
                total += eps_time;
            }
            //fout.open("Util_" + std::to_string(rank), std::ios::out);
            //fout << 0 << "\n";
            //fout.close();
            wait += elapsed(wait_st, wait_ed);
        }
        *__latency = total;
        std::cout  << req_id << " Finished @ " << total / 1000.0 << ", Wait time =" << wait / 1000000.0 << ", QoS Requirement is: " << qos_requirement << std::endl;
        if (total / 1000.0 - 15.0 < 0.0) *__is_satisfied = true;
    }
};

#endif