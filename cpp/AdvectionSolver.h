#ifndef ADVECTIONSOLVER_H
#define ADVECTIONSOLVER_H

#include <vector>

class AdvectionSolver {
public:
    AdvectionSolver(double dx, double dt, int nx);
    void initialize();
    void advance();
    void outputResults() const;

private:
    double dx; // Spatial step size
    double dt; // Time step size
    int nx; // Number of spatial points
    std::vector<double> u; // Solution vector
    double c = 1.0; // Advection speed, assuming it's constant for simplicity
};

#endif // ADVECTIONSOLVER_H