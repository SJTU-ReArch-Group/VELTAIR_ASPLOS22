#include <iostream>
#include <mpi.h>
#include "request.hpp"
using namespace std;

int main(int argc, char** argv) {
    MPI_Init(NULL, NULL);
    int nproc, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    float task_begin_time[12] = {0.0f, 500000.0f, 1000000.0f, 1500000.0, 2000000.0f, 2500000.0f, 3000000.0f, 3500000.0f, 4000000.0f, 4500000.0f, 5000000.0f, 5500000.0f};
    int total_task = 12;
    std::map<int, pair<long long, long long>> task_arrive_map;
    struct timeval basetime;
    gettimeofday(&basetime, NULL);
    if (rank == 0) {
        int task_id;
        struct timeval arr, cur, worker_active;
        char buffer[2], long_buffer[20];
        for (int i = 0; i < total_task; i++) {
            gettimeofday(&cur, NULL);
            int sleep_time = max(0, (int)(task_begin_time[i] - elapsed(basetime, cur)));
            usleep(sleep_time);
            gettimeofday(&arr, NULL);
            task_arrive_map[i] = pair<long long, long long>(arr.tv_sec, arr.tv_usec);
            int worker_idx = -1;
SPIN_CHECK:
            for (int w = 1; w <= WORKER_NUM; w++) {
                ifstream fin("IsRunning_" + to_string(w), ios::in);
                fin.getline(buffer, 2);
                fin.close();
                if (buffer[0] == '0') {
                    worker_idx = w;
                    break;
                }
            }
            if (worker_idx == -1) {
                goto SPIN_CHECK;
            }
            else {
                MPI_Request req;
                // gettimeofday(&worker_active, NULL);
                // cout << "Worker: " << worker_idx << " has spare slot @ " << elapsed(basetime,  worker_active) << endl;
                ofstream fout("IsRunning_" + to_string(worker_idx), ios::out);
                fout << 1 << endl;
                fout.close();
                ifstream fin("Finished_" + to_string(worker_idx), ios::in);
                fin.getline(long_buffer, 20);
                fin.close();
                int finished_tasks = atoi(long_buffer);
                cout << "Worker " << worker_idx << " has " << finished_tasks << " finished tasks" << endl;
                task_id = i;
                MPI_Isend(&task_id, 1, MPI_INT, worker_idx, finished_tasks, MPI_COMM_WORLD, &req);
            }
            // Sleep for a time, related with poisson distribution
        }
        ofstream fout_terminate("Terminate", ios::out);
        fout_terminate << 1 << endl;
        fout_terminate.close();
    }
    else {
        /*
        task resnet50_task("ResNet-50", "../configs/ResNet50.config", "../prof_output/ResNet50.out");
        task googlenet_task("GoogLeNet", "../configs/GoogLeNet.config", "../prof_output/GoogLeNet.out");
        request req_0(resnet50_task, rank, 0, 0);
        req_0.requestCompile();
        req_0.requestSchedule();
        request req_1(resnet50_task, rank, 0, 0);
        req_1.requestCompile();
        req_1.requestSchedule();
        */
        bool is_satisfied = false;
        float latency, overhead;
        int task_id, finished_tasks;
        MPI_Status _st;
        struct timeval ed, overhead_st, overhead_ed;
        char buffer[2], long_buffer[20];
        while (true) {
            ifstream fin("IsRunning_" + to_string(rank), ios::in);
            fin.getline(buffer, 2);
            fin.close();
            if (buffer[0] == '1') {
                // gettimeofday(&overhead_st, NULL);
                ifstream fin("Finished_" + to_string(rank), ios::in);
                fin.getline(long_buffer, 20);
                finished_tasks = atoi(long_buffer);
                fin.close();
                MPI_Recv(&task_id, 1, MPI_INT, 0, finished_tasks, MPI_COMM_WORLD, &_st);
                // gettimeofday(&overhead_ed, NULL);

                // req_0.requestExecute(&is_satisfied, &latency, rank);
                usleep(590000);
                // usleep(3000000);
                gettimeofday(&ed, NULL);
                ofstream fout_running("IsRunning_" + to_string(rank), ios::out);
                fout_running << 0 << endl;
                fout_running.close();
                ofstream fout_finished("Finished_" + to_string(rank), ios::out);
                fout_finished << finished_tasks + 1 << "\n";
                // cout << rank << " " << finished_tasks + 1 << " " << to_string(finished_tasks + 1).c_str() << endl;
                fout_finished.close();
                ofstream fout_time("Time_" + to_string(rank), ios::out|ios::app);
                fout_time << task_id << ", " << ed.tv_sec << ", " << ed.tv_usec << "\n";
                fout_time.close();
            }
            ifstream fin_terminate("Terminate", ios::in);
            fin_terminate.getline(buffer, 2);
            fin_terminate.close();
            if (buffer[0] == '1') {
                break;
            }
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        char long_buffer[256];
        ofstream fout_terminate("Terminate", ios::out);
        fout_terminate << 0 << "\n";
        fout_terminate.close();
        for (int w = 1; w <= WORKER_NUM; w++) {
            ifstream fin_time("Time_" + to_string(w), ios::in);
            while (!fin_time.eof()) {
                fin_time.getline(long_buffer, 256);
                string task_qos_info(long_buffer);
                if (task_qos_info.size() < 10) break;
                regex filter(",");
                vector <string> task_qos_info_arr(sregex_token_iterator(task_qos_info.begin(), task_qos_info.end(), filter, -1), sregex_token_iterator());
                // cout << task_qos_info_arr[0] << " " << 1000000.0 * (atoi(task_qos_info_arr[1].c_str()) - basetime.tv_sec) + 1.0 * (atoi(task_qos_info_arr[2].c_str()) - basetime.tv_usec) - (1 + atoi(task_qos_info_arr[0].c_str())) * 1000000.0 << endl;
                cout << task_qos_info_arr[0] << " " << (1000000.0 * (atoi(task_qos_info_arr[1].c_str()) - basetime.tv_sec) + 1.0 * (atoi(task_qos_info_arr[2].c_str()) - basetime.tv_usec) - task_begin_time[atoi(task_qos_info_arr[0].c_str())]) / 100000.0 << endl;
            }
            fin_time.close();
        }
    }
    MPI_Finalize();
}