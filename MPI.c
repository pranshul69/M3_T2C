#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>

void quick_sort(int *array, int left, int right) {
    int i, j, pivot, temp;

    if (left < right) {
        pivot = left;
        i = left;
        j = right;

        while (i < j) {
            while (array[i] <= array[pivot] && i <= right)
                i++;
            while (array[j] > array[pivot])
                j--;
            if (i < j) {
                temp = array[i];
                array[i] = array[j];
                array[j] = temp;
            }
        }

        temp = array[pivot];
        array[pivot] = array[j];
        array[j] = temp;

        quick_sort(array, left, j - 1);
        quick_sort(array, j + 1, right);
    }
}

int main(int argc, char **argv) {
    int rank, size, *array, *chunk, a, x, order;
    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        printf("Enter the number of elements in the array: ");
        fflush(stdout);
        scanf("%d", &a);

        array = (int *) malloc(a * sizeof(int));
        printf("Enter the elements of the array: ");
        fflush(stdout);
        for (x = 0; x < a; x++)
            scanf("%d", &array[x]);

        quick_sort(array, 0, a - 1);
    }

    MPI_Bcast(&a, 1, MPI_INT, 0, MPI_COMM_WORLD);

    chunk = (int *) malloc(a / size * sizeof(int));
    MPI_Scatter(array, a / size, MPI_INT, chunk, a / size, MPI_INT, 0, MPI_COMM_WORLD);

    clock_t start_time = clock();
    quick_sort(chunk, 0, a / size - 1);
    clock_t end_time = clock();

    double duration = (double)(end_time - start_time) / ((double)CLOCKS_PER_SEC / 1000000); // cast to double before computing duration
    if (rank == 0) {
        double duration = (double)(end_time - start_time) / ((double)CLOCKS_PER_SEC / 1000000); // cast to double before computing duration
        printf("Execution time is %f microseconds\n", duration);
    }
    

    for (order = 1; order < size; order *= 2) {
        if (rank % (2 * order) != 0) {
            MPI_Send(chunk, a / size, MPI_INT, rank - order, 0, MPI_COMM_WORLD);
            break;
        }
        int recv_size = a / size;
        if ((rank + order) < size)
            recv_size = a / size;
        else
            recv_size = a - (rank + order) * (a / size);

        int *other = (int *) malloc(recv_size * sizeof(int));
        MPI_Recv(other, recv_size, MPI_INT, rank + order, 0, MPI_COMM_WORLD, &status);

        int *temp = (int *) malloc((a / size + recv_size) * sizeof(int));
        int i = 0, j = 0, k = 0;
        while (i < a / size && j < recv_size) {
            if (chunk[i] < other[j])
                temp[k++] = chunk[i++];
            else
                temp[k++] = other[j++];
        }
        while (i < a / size)
            temp[k++] = chunk[i++];
        while (j < recv_size)
            temp[k++] = other[j++];

        free(other);
        free(chunk);
        chunk = temp;
    }

    if (rank == 0) {
        array = (int *) malloc(a * sizeof(int));
    }
    MPI_Gather(chunk, a / size, MPI_INT, array, a / size, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        for (x = 0; x < a; x++) {
            printf("%d ", array[x]);
        }
        printf("\n");
        fflush(stdout);
    }

    MPI_Finalize();
    return 0;
}