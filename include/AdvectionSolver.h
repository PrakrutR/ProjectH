#ifndef ADVECTIONSOLVER_H
#define ADVECTIONSOLVER_H

#include <vector>
#include <functional>

class AdvectionSolver {
public:
    AdvectionSolver(double dx, double dt, int nx, std::function<double(double)> initCondition);
    void initialize();
    void advance();
    std::vector<double> getResults() const;

private:
    double dx; // Spatial step size
    double dt; // Time step size
    int nx; // Number of spatial points
    std::vector<double> u; // Solution vector
    double c = 1.0; // Advection speed
    std::function<double(double)> initCondition; // Initial condition function
};

#endif // ADVECTIONSOLVER_H