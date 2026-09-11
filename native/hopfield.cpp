// Continuous three-neuron Hopfield model used by da Silva and Rech (2022).
// No discretized-map interpretation: RK4 approximates the continuous flow.
// Compile on macOS: clang++ -O3 -std=c++17 -dynamiclib -fPIC \
//     native/hopfield.cpp -o /private/tmp/libhopfield.dylib
// On Linux substitute -shared and a .so output.
#include <algorithm>
#include <cmath>
#include <limits>

namespace {
constexpr int DIM = 3;
constexpr int NSTATS = 20;
constexpr double ALIGNMENT_TIME = 100.0;

struct Model {
    double alpha, gamma;

    void rhs(const double* x, double* f) const {
        const double a = std::tanh(x[0]);
        const double b = std::tanh(x[1]);
        const double c = std::tanh(x[2]);
        f[0] = -x[0] + 1.5*a + 2.9*b + alpha*c;
        f[1] = -x[1] - 1.5*a + 1.18*b;
        f[2] = -x[2] + gamma*a - 22.0*b + 0.47*c;
    }

    void augmented_rhs(const double* y, double* f) const {
        const double a = std::tanh(y[0]);
        const double b = std::tanh(y[1]);
        const double c = std::tanh(y[2]);
        f[0] = -y[0] + 1.5*a + 2.9*b + alpha*c;
        f[1] = -y[1] - 1.5*a + 1.18*b;
        f[2] = -y[2] + gamma*a - 22.0*b + 0.47*c;
        const double p = (1.0-a*a)*y[3];
        const double q = (1.0-b*b)*y[4];
        const double r = (1.0-c*c)*y[5];
        f[3] = -y[3] + 1.5*p + 2.9*q + alpha*r;
        f[4] = -y[4] - 1.5*p + 1.18*q;
        f[5] = -y[5] + gamma*p - 22.0*q + 0.47*r;
    }
};

// f is known at entry and replaced by the derivative at the new endpoint.
// Reusing it keeps each accepted step to four new RHS evaluations.
template<int N, typename RHS>
void rk4(double* y, double* f, double dt, const RHS& rhs) {
    double trial[N], k2[N], k3[N], k4[N];
    for (int j=0; j<N; ++j) trial[j] = y[j] + 0.5*dt*f[j];
    rhs(trial, k2);
    for (int j=0; j<N; ++j) trial[j] = y[j] + 0.5*dt*k2[j];
    rhs(trial, k3);
    for (int j=0; j<N; ++j) trial[j] = y[j] + dt*k3[j];
    rhs(trial, k4);
    for (int j=0; j<N; ++j)
        y[j] += (dt/6.0)*(f[j] + 2.0*k2[j] + 2.0*k3[j] + k4[j]);
    rhs(y, f);
}

bool finite_vector(const double* y, int n) {
    for (int j=0; j<n; ++j) if (!std::isfinite(y[j])) return false;
    return true;
}

// Cubic Hermite interpolant between the two RK4 endpoints. The event is
// its x3 derivative zero, bracketed by an actual endpoint RHS sign change.
double maximum_fraction(double x0, double x1, double f0, double f1, double dt) {
    const double a = 2.0*x0 - 2.0*x1 + dt*(f0+f1);
    const double b = -3.0*x0 + 3.0*x1 - dt*(2.0*f0+f1);
    const double c = dt*f0;
    double lo = 0.0, hi = 1.0;
    for (int k=0; k<40; ++k) {
        const double mid = (lo+hi)*0.5;
        if ((3.0*a*mid+2.0*b)*mid+c > 0.0) lo=mid;
        else hi=mid;
    }
    return (lo+hi)*0.5;
}

void event_state(double s, const double* x0, const double* x1,
                 const double* f0, const double* f1, double dt, double* result) {
    const double s2=s*s, s3=s2*s;
    for (int j=0; j<DIM; ++j)
        result[j] = (2*s3-3*s2+1)*x0[j] + (s3-2*s2+s)*dt*f0[j]
                  + (-2*s3+3*s2)*x1[j] + (s3-s2)*dt*f1[j];
}

struct Moments {
    double mean[6] = {}, m2[6] = {}, speed2_mean = 0.0;
    int count = 0;
    void update(const double* x, const double* f) {
        ++count;
        double values[6] = {x[0], x[1], x[2], std::tanh(x[0]),
                            std::tanh(x[1]), std::tanh(x[2])};
        for (int j=0; j<6; ++j) {
            const double delta=values[j]-mean[j];
            mean[j] += delta/count;
            m2[j] += delta*(values[j]-mean[j]);
        }
        const double speed2=f[0]*f[0]+f[1]*f[1]+f[2]*f[2];
        speed2_mean += (speed2-speed2_mean)/count;
    }
};
} // namespace

extern "C" {

// Both helper outputs use ordinary row-major layout.
void hopfield_rhs(double alpha, double gamma, const double* state, double* rhs) {
    Model{alpha,gamma}.rhs(state,rhs);
}

void hopfield_jacobian(double alpha, double gamma, const double* state, double* jac) {
    double a=std::tanh(state[0]), b=std::tanh(state[1]), c=std::tanh(state[2]);
    a=1-a*a; b=1-b*b; c=1-c*c;
    jac[0]=-1+1.5*a; jac[1]=2.9*b; jac[2]=alpha*c;
    jac[3]=-1.5*a; jac[4]=-1+1.18*b; jac[5]=0;
    jac[6]=gamma*a; jac[7]=-22*b; jac[8]=-1+0.47*c;
}

// Thread safe: all mutable state belongs to this call.
//
// events: capacity x 4 doubles, rows (time,x1,x2,x3), only the returned
// number of rows are valid. The most recent capacity x3 local maxima are
// returned in chronological order. Times start at the END of burn-in.
// capacity=0 is supported with events=nullptr for integration-only checks.
//
// stats: 20 doubles with this ABI:
// [0] total detected peaks (including those overwritten by the ring buffer)
// [1] largest one-vector finite-time Lyapunov estimate per unit physical time
// [2] duration used by that estimate, excluding 100 time units of alignment
// [3] completed sampling duration
// [4:7] mean state; [7:10] population standard deviation of state
// [10:13] mean tanh(state); [13:16] its population standard deviation
// [16] mean squared speed; [17] terminal speed
// [18] completed burn steps; [19] completed sampling steps
// Moments are unweighted samples at every completed measurement endpoint.
// FTLE is NaN if no time remains after alignment. A periodic autonomous
// flow has a neutral tangent direction: near-zero FTLE is not instability.
//
// Returns retained peak count, or -1 invalid arguments / -2 nonfinite
// integration or degenerate tangent. On -2 finalstate and stats describe
// progress; callers must preserve failure status, not classify the cell.
int hopfield_integrate(double alpha, double gamma, const double* initial,
                       double dt, int burn_steps, int sample_steps, int capacity,
                       double* finalstate, double* events, double* stats) {
    if (!initial || !finalstate || !stats || (capacity>0 && !events) ||
        !std::isfinite(alpha) || !std::isfinite(gamma) || !std::isfinite(dt) ||
        dt<=0.0 || burn_steps<0 || sample_steps<0 || capacity<0 ||
        !finite_vector(initial,3)) return -1;
    std::fill(stats,stats+NSTATS,std::numeric_limits<double>::quiet_NaN());
    double y[6]={initial[0],initial[1],initial[2],0,0,0}, f[6]={};
    Model model{alpha,gamma};
    const auto state_rhs=[&model](const double* x,double* dx){model.rhs(x,dx);};
    const auto aug_rhs=[&model](const double* x,double* dx){model.augmented_rhs(x,dx);};
    model.rhs(y,f);
    int burn_done=0, sample_done=0, peaks=0, next_slot=0, last_norm_step=0;
    double log_growth=0.0, exponent_time=0.0;
    bool failed=!finite_vector(f,3);
    for (int i=0; i<burn_steps && !failed; ++i) {
        rk4<3>(y,f,dt,state_rhs);
        ++burn_done;
        failed=!finite_vector(y,3) || !finite_vector(f,3);
    }
    y[3]=1/std::sqrt(6.0); y[4]=std::sqrt(2.0/6.0); y[5]=std::sqrt(3.0/6.0);
    model.augmented_rhs(y,f);
    Moments moments;
    // A double threshold avoids overflow for otherwise valid tiny dt.
    const double alignment_steps=std::ceil(ALIGNMENT_TIME/dt);
    for (int i=1; i<=sample_steps && !failed; ++i) {
        double previous_y[3]={y[0],y[1],y[2]};
        double previous_f[3]={f[0],f[1],f[2]};
        rk4<6>(y,f,dt,aug_rhs);
        sample_done=i;
        failed=!finite_vector(y,6) || !finite_vector(f,6);
        if (failed) break;
        moments.update(y,f);
        if (previous_f[2]>0.0 && f[2]<=0.0) {
            ++peaks;
            if (capacity>0) {
                const double fraction=maximum_fraction(previous_y[2],y[2],
                                                       previous_f[2],f[2],dt);
                double* row=events+4*next_slot;
                row[0]=(i-1+fraction)*dt;
                event_state(fraction,previous_y,y,previous_f,f,dt,row+1);
                next_slot=(next_slot+1)%capacity;
            }
        }
        if (i%10==0 || i==sample_steps || i==alignment_steps) {
            const double norm=std::hypot(y[3],y[4],y[5]);
            if (!std::isfinite(norm) || norm<=0.0) {failed=true; break;}
            if (last_norm_step>=alignment_steps) {
                log_growth+=std::log(norm);
                exponent_time+=(i-last_norm_step)*dt;
            }
            for (int j=3; j<6; ++j) {y[j]/=norm; f[j]/=norm;}
            last_norm_step=i;
        }
    }
    const int retained=std::min(peaks,capacity);
    if (peaks>capacity && capacity>0)
        std::rotate(events,events+4*next_slot,events+4*capacity);
    std::copy(y,y+3,finalstate);
    stats[0]=peaks;
    stats[1]=exponent_time>0.0 ? log_growth/exponent_time
                              : std::numeric_limits<double>::quiet_NaN();
    stats[2]=exponent_time; stats[3]=sample_done*dt;
    if (moments.count>0) {
        for (int j=0; j<3; ++j) {
            stats[4+j]=moments.mean[j];
            stats[7+j]=std::sqrt(std::max(0.0,moments.m2[j]/moments.count));
            stats[10+j]=moments.mean[j+3];
            stats[13+j]=std::sqrt(std::max(0.0,moments.m2[j+3]/moments.count));
        }
        stats[16]=moments.speed2_mean;
    }
    stats[17]=std::hypot(f[0],f[1],f[2]);
    stats[18]=burn_done; stats[19]=sample_done;
    return failed ? -2 : retained;
}
} // extern C
