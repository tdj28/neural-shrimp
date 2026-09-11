// Additional trace ABI. The frozen model and RK4 implementation remain intact.
// Build this translation unit by itself; it includes the original implementation.
#include "hopfield.cpp"

extern "C" {
// states: capacity x 3, including initial state at row zero. Require steps to
// be a multiple of stride and capacity == steps/stride + 1, so both endpoints
// are retained. progress: (completed RK4 steps, retained rows), including on
// integration failure. finalstate retains the most recent attempted state.
// Return retained rows, -1 invalid input, or -2 nonfinite integration.
int hopfield_trace(double alpha, double gamma, const double* initial,
                   double dt, int steps, int stride, int capacity,
                   double* states, double* finalstate, double* progress) {
    if (!initial || !states || !finalstate || !progress ||
        !std::isfinite(alpha) || !std::isfinite(gamma) ||
        !std::isfinite(dt) || dt <= 0.0 || steps < 0 || stride <= 0 ||
        steps % stride != 0 || capacity <= 0 ||
        capacity != steps / stride + 1 || !finite_vector(initial, 3)) return -1;
    std::fill(states, states + 3LL*capacity,
              std::numeric_limits<double>::quiet_NaN());
    std::copy(initial, initial+3, states);
    double y[3]={initial[0],initial[1],initial[2]}, f[3]={};
    Model model{alpha,gamma};
    const auto rhs=[&model](const double* x,double* dx){model.rhs(x,dx);};
    model.rhs(y,f);
    int done=0, retained=1;
    bool failed=!finite_vector(f,3);
    for (int step=1; step<=steps && !failed; ++step) {
        rk4<3>(y,f,dt,rhs);
        done=step;
        failed=!finite_vector(y,3) || !finite_vector(f,3);
        if (failed) break;
        if (step % stride == 0) {
            std::copy(y,y+3,states+3LL*retained);
            ++retained;
        }
    }
    std::copy(y,y+3,finalstate);
    progress[0]=done;
    progress[1]=retained;
    return failed ? -2 : retained;
}
}
