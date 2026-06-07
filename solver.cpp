#include "solver.h"
#include <algorithm>

namespace gacha 
{

Solution solve_objectiveB(const Params& prm) 
{
    const int B = prm.B;
    Solution sol;
    sol.prm = prm;

    std::array<double, 2> zerod = {{0.0, 0.0}};
    std::array<Action, 2> saves = {{SAVE, SAVE}};
    sol.V.assign(B + 1, std::vector<std::array<double, 2> >(N, zerod));
    sol.policy.assign(B + 1, std::vector<std::array<Action, 2> >(N, saves));

    // Boundary: V*(0,k,f) = 0 and pi*(0,k,f) = SAVE (set by the init above).

    // Backward induction on b: q(b,.,.) depends only on V*(b-1,.,.).
    for (int b = 1; b <= B; ++b) 
    {
        for (int k = 0; k < N; ++k) 
        {
            const double p = pity_prob(k);
            const int knext = std::min(k + 1, N - 1); 
            for (int f = 0; f <= 1; ++f) 
            {
                const double q_save = 0.0; //if SAVE, can't get anything or lose anything. 
                double hit_value; //this is a middle vairable. 
                if (f == 1) hit_value = prm.W;
                else hit_value = prm.q * prm.W + (1.0 - prm.q) * sol.V[b - 1][0][1]; //the f is changed to 1 if missed. 
                const double miss_value = sol.V[b - 1][knext][f]; // miss: f kept
                const double q_pull = -prm.cost
                                    + p * hit_value
                                    + (1.0 - p) * miss_value;

                // V*(s) = max(q(s,SAVE), q(s,PULL)); pi*(s) = argmax 
                if (q_pull > q_save) 
                {
                    sol.V[b][k][f]      = q_pull;
                    sol.policy[b][k][f] = PULL;
                } 
                else 
                {
                    sol.V[b][k][f]      = q_save;
                    sol.policy[b][k][f] = SAVE;
                }
            }
        }
    }
    return sol;
}

PolicyEval evaluate_policy(const Solution& sol) {
    const int B = sol.prm.B;
    const double q = sol.prm.q;

    PolicyEval e;
    std::array<double, 2> zerod = {{0.0, 0.0}};
    e.p_featured.assign(B + 1, std::vector<std::array<double, 2> >(N, zerod));
    e.exp_pulls.assign(B + 1, std::vector<std::array<double, 2> >(N, zerod));

    // Same backward recursion, but EVALUATING the fixed policy pi* (no max):
    // expectations of P(featured) and pulls-spent under pi*.
    for (int b = 1; b <= B; ++b) {
        for (int k = 0; k < N; ++k) {
            const double p     = pity_prob(k);
            const int    knext = std::min(k + 1, N - 1);
            for (int f = 0; f <= 1; ++f) {
                if (sol.policy[b][k][f] == SAVE) {
                    e.p_featured[b][k][f] = 0.0; // pi* stops here -> never wins
                    e.exp_pulls[b][k][f]  = 0.0;
                    continue;
                }
                // pi* says PULL: take one pull, then recurse on b-1 states.
                double pf, ep;
                if (f == 1) {
                    pf = p * 1.0 + (1.0 - p) * e.p_featured[b - 1][knext][1];
                    ep = 1.0      + (1.0 - p) * e.exp_pulls[b - 1][knext][1];
                } else {
                    const double hit_pf = q * 1.0
                                        + (1.0 - q) * e.p_featured[b - 1][0][1];
                    const double hit_ep = q * 0.0
                                        + (1.0 - q) * e.exp_pulls[b - 1][0][1];
                    pf = p * hit_pf + (1.0 - p) * e.p_featured[b - 1][knext][0];
                    ep = 1.0 + p * hit_ep + (1.0 - p) * e.exp_pulls[b - 1][knext][0];
                }
                e.p_featured[b][k][f] = pf;
                e.exp_pulls[b][k][f]  = ep;
            }
        }
    }
    return e;
}

} // namespace gacha
