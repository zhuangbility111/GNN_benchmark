#include "utils.h"

int divup(int x, int y) { return (x + y - 1) / y; }


void divide_work(int* work_range, int total_work, int num_threads) {
    int chunk_size;
    int remain_work = total_work;
    work_range[0] = 0;
    for (int i = 0; i < num_threads; i++) {
        chunk_size = divup(remain_work, num_threads - i);
        work_range[i + 1] = work_range[i] + chunk_size;
        remain_work -= chunk_size;
    }
    work_range[num_threads] = total_work;
}
