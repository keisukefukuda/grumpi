#include <iostream>

#include <mpi.h>

#include "grumpi/grumpi.hpp"

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int size = grumpi::Comm_size();
    int rank = grumpi::Comm_rank();
    int intra_rank;
    
    grumpi::Comm_local_rank(MPI_COMM_WORLD, &intra_rank);

    for (int i = 0; i < size; i++) {
        if (rank == i) {
            std::cout << "Rank " << i << " " << intra_rank << std::endl;
            std::cout.flush();
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}
