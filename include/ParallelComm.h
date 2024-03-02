#ifndef PARALLEL_COMM_H
#define PARALLEL_COMM_H

#include <mpi.h>

// Initializes the MPI environment. Should be called at the start of the program.
void initParallelEnvironment(int *argc, char ***argv);

// Finalizes the MPI environment. Should be called at the end of the program.
void finalizeParallelEnvironment();

// Performs halo exchange for a 2D domain decomposition.
// 'data' is the local chunk of a larger domain, including space for halo regions on all sides.
// 'nx' and 'ny' are the dimensions of the local domain including halo regions.
// 'halo_size' specifies the width of the halo region.
// The process grid dimensions are provided in 'px' and 'py', with the current process position as 'px_pos' and 'py_pos'.
void exchangeHalos2D(double *data, int nx, int ny, int halo_size, int px, int py, int px_pos, int py_pos);

#endif // PARALLEL_COMM_H