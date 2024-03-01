#include "AdvectionSolver.h"
#include <iostream>
#include <fstream>

AdvectionSolver::AdvectionSolver(double dx, double dt, int nx) : dx(dx), dt(dt), nx(nx), u(nx, 0.0) {}

void AdvectionSolver::initialize() {
    // Simple initial condition: Gaussian profile
    double x0 = 0.5 * nx * dx; // Center
    double sigma = 0.05; // Width of the Gaussian
    for (int i = 0; i < nx; ++i) {
        double x = i * dx;
        u[i] = exp(-pow((x - x0) / sigma, 2));
    }
}

void AdvectionSolver::advance() {
    // Use a simple finite difference scheme for advection
    std::vector<double> u_new(nx, 0.0);
    for (int i = 1; i < nx - 1; ++i) {
        u_new[i] = u[i] - c * dt / dx * (u[i] - u[i - 1]);
    }
    u = u_new;
}

void AdvectionSolver::outputResults() const {
    std::ofstream outFile("results.dat");
    for (int i = 0; i < nx; ++i) {
        outFile << i * dx << " " << u[i] << std::endl;
    }
}

int main() {
    double dx = 0.01;
    double dt = 0.005;
    int nx = 100;
    AdvectionSolver solver(dx, dt, nx);

    solver.initialize();
    for (int t = 0; t < 100; ++t) { // Run for 100 time steps
        solver.advance();
    }
    solver.outputResults();

    return 0;
}