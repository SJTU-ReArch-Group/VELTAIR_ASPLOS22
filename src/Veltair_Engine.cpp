#include <iostream>
#include <mpi.h>
#include "request.hpp"
using namespace std;

int main(int argc, char** argv) {
    MPI_Init(NULL, NULL);
    int nproc, rank, core_number;
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    cpu_set_t local_mask;
    int is_task_remained = 1, selected_worker = -1;
    float timeout = 10.0f;
    int total_task = 320;
    int local_satisfied = 0;
    std::vector <float> local_latency;
    local_latency.clear();
    int rank_finished = 0;
    int core_available = sysconf(_SC_NPROCESSORS_ONLN);
    int queue_len[WORKER_NUM];
    for (int i = 0; i < WORKER_NUM; i++) {
        queue_len[i] = 0;
    }
    struct timeval st, ed;
    gettimeofday(&st, NULL);
    if (rank == 0) {
        // Workload dispatcher.
        CPU_ZERO(&local_mask);
        CPU_SET(63, &local_mask);
        // sched_setaffinity(0, sizeof(cpu_set_t), &local_mask);

        size_t task_num = total_task, launched_task = 0;
        float poi_k = 0.5, lambda = 2.0;
        int inteval;
        // task vgg19_task("VGG-19", "../configs/VGG19.config");
        srand(time(0));
        while (task_num--) {
            inteval = 1;
            // if (task_num == 10) sleep(5);
            usleep(atoi(argv[1]));
            int target_worker, target_worker_queue_len, launched = 0, tmp;
            find_min_idx(queue_len, WORKER_NUM, &target_worker, &target_worker_queue_len, &tmp);
            for (int i = 0; i < WORKER_NUM; i++) {
                launched += queue_len[i];
            }
            queue_len[target_worker]++;
            selected_worker = target_worker;
            for (int i = 1; i < nproc; i++) {
                MPI_Send(&selected_worker, 1, MPI_INT, i, launched, MPI_COMM_WORLD);
            }
            launched_task++;
        }
        // is_task_remained = 0;
        // for (int i = 1; i < nproc; i++) {
        //     MPI_Send(&is_task_remained, 1, MPI_INT, i, launched_task, MPI_COMM_WORLD);
        // }
    }
    else {

        // task vgg19_task("VGG-19", "../configs/VGG19.config");
        // task vgg19_ksat("VGG-19", "../configs/VGG19.config");
        task resnet50_task("ResNet-50", "../configs/ResNet50.config", "../prof_output/ResNet50.out");
        task googlenet_task("GoogLeNet", "../configs/GoogLeNet.config", "../prof_output/GoogLeNet.out");
        task efficientnet_task("EfficientNet", "../configs/Bert.config", "../prof_output/EfficientNet.out");
        // task background_task("Background", "../configs/Background.config", "../prof_output/Background.config");
        // ßcout << "Task Set" << endl;
        CPU_ZERO(&local_mask);
        for (int i = 0; i < 64; i++) CPU_SET(i, &local_mask);
        sched_setaffinity(0, sizeof(cpu_set_t), &local_mask);
        vector <request*> task_queue;
        int flip = 0;
        MPI_Request req;
        MPI_Status st;
        MPI_Irecv(&is_task_remained, 1, MPI_INT, 0, total_task, MPI_COMM_WORLD, &req);
        while (true) {
            int launched = 0;
            for (int i = 0; i < WORKER_NUM; i++) {
                launched += queue_len[i];
            }
            // std::cout << "Receiving: " << launched << std::endl;
            if (launched == total_task) break;
            MPI_Recv(&selected_worker, 1, MPI_INT, 0, launched, MPI_COMM_WORLD, &st);
            // std::cout << "Received" << std::endl;
            queue_len[selected_worker]++;
            bool is_satisfied = false;
            float latency;
            request req_00(resnet50_task, rank, queue_len[selected_worker], launched);
            req_00.requestCompile();
            req_00.requestSchedule(atoi(argv[2]));
            req_00.requestExecute(&is_satisfied, &latency, rank);
            // if (selected_worker == rank - 1) {
            //     int core_use = min(core_available, 10); // Can be detemined via some kind of scheduling strategy?
            //     // std::cout << rank << " " << launched << std::endl;
            //     if (rank == 1) {
            //         if (flip == 0) {
            //             request req_00(efficientnet_task, rank, queue_len[selected_worker], launched);
            //             req_00.requestCompile();
            //             req_00.requestSchedule(atoi(argv[2]));
            //             req_00.requestExecute(&is_satisfied, &latency, rank);
            //             if (is_satisfied) local_satisfied++;
            //             // flip = 1;;
            //         }
            //         else {
            //             request req_01(googlenet_task, rank, queue_len[selected_worker], launched);
            //             req_01.requestCompile();
            //             req_01.requestSchedule();
            //             req_01.requestExecute(&is_satisfied, &latency, rank);
            //             if (is_satisfied) local_satisfied++;
            //             flip = 0;
            //         }
            //     }
            //     else if (rank == 2) {
            //         if (flip == 0) {
            //             request req_1(efficientnet_task, rank, queue_len[selected_worker], launched);
            //             req_1.requestCompile();
            //             req_1.requestSchedule();
            //             req_1.requestExecute(&is_satisfied, &latency, rank);
            //             if (is_satisfied) local_satisfied++;
            //             // flip = 1;;
            //         }
            //         else {
            //             request req_1(googlenet_task, rank, queue_len[selected_worker], launched);
            //             req_1.requestCompile();
            //             req_1.requestSchedule();
            //             req_1.requestExecute(&is_satisfied, &latency, rank);
            //             if (is_satisfied) local_satisfied++;
            //             flip = 0;
            //         }
            //     }
                
            //     else if (rank == 3) {
            //         if (flip == 0) {
            //             request req_2(efficientnet_task, rank, queue_len[selected_worker], launched);
            //             req_2.requestCompile();
            //             req_2.requestSchedule();
            //             req_2.requestExecute(&is_satisfied, &latency, rank);
            //             if (is_satisfied) local_satisfied++;
            //             // flip = 1;;
            //         }
            //         else {
            //             request req_2(googlenet_task, rank, queue_len[selected_worker], launched);
            //             req_2.requestCompile();
            //             req_2.requestSchedule();
            //             req_2.requestExecute(&is_satisfied, &latency, rank);
            //             if (is_satisfied) local_satisfied++;
            //             flip = 0;
            //         }
            //     }
            //     else if (rank == 4) {
            //         if (flip == 0) {
            //             request req_3(efficientnet_task, rank, queue_len[selected_worker], launched);
            //             req_3.requestCompile();
            //             req_3.requestSchedule();
            //             req_3.requestExecute(&is_satisfied, &latency, rank);
            //             if (is_satisfied) local_satisfied++;
            //             // flip = 1;;
            //         }
            //         else {
            //             request req_3(googlenet_task, rank, queue_len[selected_worker], launched);
            //             req_3.requestCompile();
            //             req_3.requestSchedule();
            //             req_3.requestExecute(&is_satisfied, &latency, rank);
            //             if (is_satisfied) local_satisfied++;
            //             flip = 0;
            //         }
            //     }
            //     else if (rank == 5) {
            //         if (flip == 0) {
            //             request req_4(efficientnet_task, rank, queue_len[selected_worker], launched);
            //             req_4.requestCompile();
            //             req_4.requestSchedule();
            //             req_4.requestExecute(&is_satisfied, &latency, rank);
            //             if (is_satisfied) local_satisfied++;
            //             // flip = 1;;
            //         }
            //         else {
            //             request req_4(googlenet_task, rank, queue_len[selected_worker], launched);
            //             req_4.requestCompile();
            //             req_4.requestSchedule();
            //             req_4.requestExecute(&is_satisfied, &latency, rank);
            //             if (is_satisfied) local_satisfied++;
            //             flip = 0;
            //         }
            //     }
            //     else if (rank == 6) {
            //         if (flip == 0) {
            //             request req_5(efficientnet_task, rank, queue_len[selected_worker], launched);
            //             req_5.requestCompile();
            //             req_5.requestSchedule();
            //             req_5.requestExecute(&is_satisfied, &latency, rank);
            //             if (is_satisfied) local_satisfied++;
            //             // flip = 1;;
            //         }
            //         else {
            //             request req_5(efficientnet_task, rank, queue_len[selected_worker], launched);
            //             req_5.requestCompile();
            //             req_5.requestSchedule();
            //             req_5.requestExecute(&is_satisfied, &latency, rank);
            //             if (is_satisfied) local_satisfied++;
            //             flip = 0;
            //         }
            //     }
            //     else if (rank == 7) {
            //         if (flip == 0) {
            //             request req_6(efficientnet_task, rank, queue_len[selected_worker], launched);
            //             req_6.requestCompile();
            //             req_6.requestSchedule();
            //             req_6.requestExecute(&is_satisfied, &latency, rank);
            //             if (is_satisfied) local_satisfied++;
            //             // flip = 1;;
            //         }
            //         else {
            //             request req_6(googlenet_task, rank, queue_len[selected_worker], launched);
            //             req_6.requestCompile();
            //             req_6.requestSchedule();
            //             req_6.requestExecute(&is_satisfied, &latency, rank);
            //             if (is_satisfied) local_satisfied++;
            //             flip = 0;
            //         }
            //     }
            //     else if (rank == 8) {
            //         if (flip == 0) {
            //             request req_7(efficientnet_task, rank, queue_len[selected_worker], launched);
            //             req_7.requestCompile();
            //             req_7.requestSchedule();
            //             req_7.requestExecute(&is_satisfied, &latency, rank);
            //             if (is_satisfied) local_satisfied++;
            //             // flip = 1;;
            //         }
            //         else {
            //             request req_7(googlenet_task, rank, queue_len[selected_worker], launched);
            //             req_7.requestCompile();
            //             req_7.requestSchedule();
            //             req_7.requestExecute(&is_satisfied, &latency, rank);
            //             if (is_satisfied) local_satisfied++;
            //             flip = 0;
            //         }
            //     }
                
            //     local_latency.push_back(latency);
            // }
            // int is_received_flag_from_0 = 0;
            // MPI_Test(&req, &is_received_flag_from_0, MPI_STATUS_IGNORE);
            // std::cout << is_received_flag_from_0 << std::endl;
            // if (is_received_flag_from_0) {
            //     MPI_Wait(&req, MPI_STATUS_IGNORE);
            // }
        }
        std::cout << "Local Satisfied of Rank " << rank << " is " << local_satisfied << ", Mean Latency: " << mean(local_latency) << " " << min(local_latency) << " " << max(local_latency) << std::endl;
    }
    MPI_Finalize();
    return 0;
}
