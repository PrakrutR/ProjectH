#include <AdvectionSolver.h>
#include <cmath>

AdvectionSolver::AdvectionSolver(double dx, double dt, int nx, std::function<double(double)> initCondition)
    : dx(dx), dt(dt), nx(nx), u(nx, 0.0), initCondition(initCondition) {}

void AdvectionSolver::initialize() {
    for (int i = 0; i < nx; ++i) {
        double x = i * dx;
        u[i] = initCondition(x);
    }
}

void AdvectionSolver::advance() {
    std::vector<double> u_new(nx, 0.0);
    for (int i = 1; i < nx - 1; ++i) {
        u_new[i] = u[i] - c * dt / dx * (u[i] - u[i - 1]);
    }
    u = u_new;
}

std::vector<double> AdvectionSolver::getResults() const {
    return u;
}