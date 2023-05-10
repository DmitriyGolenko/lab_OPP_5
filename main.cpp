#include <mpi.h>
#include <pthread.h>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cmath>

#define FAIL 0
#define SUCCESS 1
#define REQUEST_TAG 2
#define ANSWER_TAG 3
#define NEED_TASKS 4
#define TURN_OFF 5

#define TASKS_IN_LIST 200
#define L 1000
#define ITERATION 16

typedef struct Task{
    int repeatNum;
} Task;
typedef Task TaskList[TASKS_IN_LIST];
int processRank;
int commSize;
int iterCounter = 0;
int curTask = 0;
int listSize = 0;
double globalRes = 0;
double SummaryDisbalance = 0;
TaskList taskList;
pthread_mutex_t mutex;

void getTaskList() {
    listSize = TASKS_IN_LIST;
    for (int i = processRank * TASKS_IN_LIST; i < (processRank + 1) * TASKS_IN_LIST; i++) {
        taskList[i % TASKS_IN_LIST].repeatNum = abs(TASKS_IN_LIST / 2 - i % TASKS_IN_LIST) *abs(processRank - (iterCounter % commSize)) * L;
    }
}

int getTaskFrom(int from) {
    int flag = NEED_TASKS;
    MPI_Send(&flag, 1, MPI_INT, from, REQUEST_TAG, MPI_COMM_WORLD);
    MPI_Recv(&flag, 1, MPI_INT, from, ANSWER_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    if (flag == FAIL) return FAIL;
    Task recvTask;
    MPI_Recv(&recvTask, 1, MPI_INT, from, ANSWER_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    pthread_mutex_lock(&mutex);
    taskList[listSize] = recvTask;
    listSize++;
    pthread_mutex_unlock(&mutex);
    return SUCCESS;
}

void doTask(Task task) {
    for (int i = 0; i < task.repeatNum; i++) {
        globalRes += sin(i);
    }
}


void *taskSenderThread(void *args) {
    int flag;
    while (ITERATION > iterCounter) {
        MPI_Status status;
        MPI_Recv(&flag, 1, MPI_INT, MPI_ANY_SOURCE, REQUEST_TAG, MPI_COMM_WORLD,&status);
        if (flag == TURN_OFF) break;
        pthread_mutex_lock(&mutex);
        if (curTask >= listSize - 1) {
            pthread_mutex_unlock(&mutex);
            flag = FAIL;
            MPI_Send(&flag, 1, MPI_INT, status.MPI_SOURCE, ANSWER_TAG, MPI_COMM_WORLD);
            continue;
        }
        listSize--;
        Task sendTask = taskList[listSize];
        pthread_mutex_unlock(&mutex);
        flag = SUCCESS;
        MPI_Send(&flag, 1, MPI_INT, status.MPI_SOURCE, ANSWER_TAG, MPI_COMM_WORLD);
        MPI_Send(&sendTask, 1, MPI_INT, status.MPI_SOURCE, ANSWER_TAG, MPI_COMM_WORLD);
    }
    return nullptr;
}


void *taskExecutorThread(void *args) {
    MPI_Barrier(MPI_COMM_WORLD);
    struct timespec start{}, end{};
    iterCounter = 0;
    while (iterCounter < ITERATION) {
        int tasksDone = 0;
        int hasTasks = 1;
        pthread_mutex_lock(&mutex);
        curTask = 0;
        getTaskList();
        pthread_mutex_unlock(&mutex);
        clock_gettime(CLOCK_MONOTONIC, &start);
        while (hasTasks) {
            pthread_mutex_lock(&mutex);
            if (curTask < listSize) {
                Task task = taskList[curTask];
                pthread_mutex_unlock(&mutex);
                doTask(task);
                tasksDone++;
                pthread_mutex_lock(&mutex);
                curTask++;
                pthread_mutex_unlock(&mutex);
                continue;
            }
            curTask = 0;
            listSize = 0;
            pthread_mutex_unlock(&mutex);
            hasTasks = 0;
            for (int i = 0; i < commSize; i++) {
                if (i == processRank) continue;
                if (getTaskFrom(i) == SUCCESS) {
                    hasTasks = 1;
                }
            }
        }
        clock_gettime(CLOCK_MONOTONIC, &end);
        double timeTaken = end.tv_sec - start.tv_sec + 0.000000001 * (double) (end.tv_nsec - start.tv_nsec);
        double minTime, maxTime;
        MPI_Reduce(&timeTaken, &minTime, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
        MPI_Reduce(&timeTaken, &maxTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        pthread_mutex_lock(&mutex);
        SummaryDisbalance += (maxTime - minTime) / maxTime;
        pthread_mutex_unlock(&mutex);
        MPI_Barrier(MPI_COMM_WORLD);
        iterCounter++;
    }
    int flag = TURN_OFF;
    MPI_Send(&flag, 1, MPI_INT, processRank, REQUEST_TAG, MPI_COMM_WORLD);
    return nullptr;
}


int main(int argc, char **argv) {
    int provided;
    MPI_Init_thread(nullptr, nullptr, MPI_THREAD_MULTIPLE, &provided);
    if (provided != MPI_THREAD_MULTIPLE) {
        fprintf(stderr, "Couldn't init MPI with MPI_THREAD_MULTIPLE level support\n");
        MPI_Finalize();
        return 0;
    }
    MPI_Comm_rank(MPI_COMM_WORLD, &processRank);
    MPI_Comm_size(MPI_COMM_WORLD, &commSize);
    pthread_t threads[2];
    pthread_mutex_init(&mutex, nullptr);
    pthread_attr_t attrs;
    if (pthread_attr_init(&attrs) != 0) {
        perror("Cannot initialize attributes");
        abort();
    }
    if (pthread_attr_setdetachstate(&attrs, PTHREAD_CREATE_JOINABLE) != 0) {
        perror("Error in setting attributes");
        abort();
    }
    struct timespec startGlobal{}, endGlobal{};
    if(processRank == 0){
        clock_gettime(CLOCK_MONOTONIC, &startGlobal);
    }
    if (pthread_create(&threads[0], &attrs, taskSenderThread, nullptr) != 0 ||
    pthread_create(&threads[1], &attrs, taskExecutorThread, nullptr) != 0) {
        perror("Cannot create a thread");
        abort();
    }
    pthread_attr_destroy(&attrs);
    pthread_join(threads[1], nullptr);
    for (auto thread : threads) {
        if (pthread_join(thread, nullptr) != 0) {
            perror("Cannot join a thread");
            abort();
        }
    }
    if(processRank == 0){
        clock_gettime(CLOCK_MONOTONIC, &endGlobal);
        double timeTaken = endGlobal.tv_sec - startGlobal.tv_sec + 0.000000001 * (double)(endGlobal.tv_nsec - startGlobal.tv_nsec);
        printf("\n\n\n Summary disbalance: %lf\n", SummaryDisbalance / (ITERATION) * 100);
        printf("Global time: %lf", timeTaken);
    }
    pthread_mutex_destroy(&mutex);
    MPI_Finalize();
    return 0;
}

