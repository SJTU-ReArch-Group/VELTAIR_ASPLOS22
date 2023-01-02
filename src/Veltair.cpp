#include <iostream>
#include <mpi.h>
#include "request.hpp"
using namespace std;

/*
 * Rank 0 as task dispatcher
 */

int main(int argc, char** argv) {
    MPI_Init(NULL, NULL);
    int nproc, rank, core_number;
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    cpu_set_t global_cpu;
    CPU_ZERO(&global_cpu);
    for (int i = 0; i < sysconf(_SC_NPROCESSORS_ONLN); i++) CPU_SET(i, &global_cpu);
    sched_setaffinity(0, sizeof(cpu_set_t), &global_cpu);
    if (rank == 0) {
        task vgg19_task("VGG-19", "../configs/VGG19.config");
        request vgg19_req_0(vgg19_task);
        vgg19_req_0.requestCompile();
        vgg19_req_0.requestSchedule(2);
        MPI_Barrier(MPI_COMM_WORLD);
        vgg19_req_0.requestExecute();
    }
    else {
        task resnet18_task("ResNet-18", "../configs/ResNet18.config");
        request resnet18_req_0(resnet18_task);
        resnet18_req_0.requestCompile();
        resnet18_req_0.requestSchedule(1);
        MPI_Barrier(MPI_COMM_WORLD);
        resnet18_req_0.requestExecute();
    }
    MPI_Finalize();
    return 0;
}