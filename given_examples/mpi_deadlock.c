#include <stdio.h>
#include <string.h>
#include <mpi.h>

/* This example is an unsafe MPI program. It attempts to send messages
 * between processes in a ring like fashion, i.e., process i sends a messagge 
 * to process i+1 and receives a message from process i-1. If the call to MPI_Send
 * is blocking, each process waits for the matching call to MPI_Recv, and no 
 * process will ever reach the call to MPI_Recv. Hence the program runs into a 
 * deadlock, i.e., it hangs.
 * 
 * In the code below, the call to MPI_Ssend is explicitly synchronous so it will block.
 * If you change MPI_Ssend to MPI_Send, the code _may_ work with some implementations, 
 * but it is still an unsafe program that will likely hang.
 */

int main(int argc, char *argv[]) {

    MPI_Init(&argc, &argv);

    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Status status;

    int my_rank, comm_sz;
    MPI_Comm_rank(comm, &my_rank);
    MPI_Comm_size(comm, &comm_sz);

    char greeting[100];
    sprintf(greeting, "Hello, world, I am process %d! There are %d of us in MPI_COMM_WORLD.\n", my_rank, comm_sz);

    int dst = (my_rank+1)%comm_sz; // my_rank sends to my_rank+1
    int src = (my_rank-1+comm_sz)%comm_sz; // my_rank receives from my_rank-1

    /* the following causes a deadlock! */
    MPI_Ssend(greeting, strlen(greeting)+1, MPI_CHAR, dst, 0, comm);
    MPI_Recv(greeting, 100, MPI_CHAR, src, 0, comm, &status);
    
    printf("Rank %d received message from rank %d: %s", my_rank, src, greeting);

    MPI_Barrier(comm);

    MPI_Finalize();

    return 0;

}