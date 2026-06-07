#ifndef GACHA_SOLVER_H
#define GACHA_SOLVER_H

#include <vector>
#include <array>
#include "pity.h"

namespace gacha 
{

enum Action { SAVE = 0, PULL = 1 };

// ===========================================================================
// OBJECTIVE B as a Markov Decision Process, written in standard RL notation
// (value function V_pi, action-value function q_pi, policy pi).
//
//   State    s = (b, k, f)
//        b : budget -- pulls we may still spend,      b = 0..B
//        k : pity counter,                            k = 0..N-1   (N = 90)
//        f : guarantee flag,                          f in {0,1}
//            f = 1  <=>  the next 5-star is guaranteed to be the featured unit
//
//   Action   a in A = { SAVE, PULL }                  (PULL requires b >= 1)
//        SAVE : stop pulling on this banner (terminal)
//        PULL : pay cost c, then draw once at probability p(k)
//
//   Policy   pi(s, a) = P(choose action a in state s)
//
//   Return   G = W * 1{featured obtained}  -  c * (number of PULLs executed)
//        W : value (in pulls) of obtaining the featured unit
//        c : cost paid per PULL
//        q : 50/50 win-rate, i.e. P(a 5-star is the featured unit) when f = 0
//
//   Value function          V_pi(s)   = E_pi[ G_t | S_t = s ]
//   Action-value function   q_pi(s,a) = E_pi[ G_t | S_t = s, A_t = a ]
//   Relationship            V_pi(s)   = sum_a pi(s,a) * q_pi(s,a)
//
// ---------------------------------------------------------------------------
// ACTION-VALUES (note: the cost c is paid ONCE per PULL, never per outcome):
//
//   q(s, SAVE) = 0                                    (stop: no reward, no cost)
//
//   q((b,k,1), PULL) = -c + p(k)*W                    // hit  -> win W, STOP
//                          + (1-p(k)) * V(b-1, k+1, 1)// miss -> pity advances
//
//   q((b,k,0), PULL) = -c + p(k)*[ q*W                // hit & win 50/50 -> W, STOP
//                                + (1-q)*V(b-1,0,1) ] // hit & lose -> off-banner:
//                                                     //   guarantee turns on, k resets
//                          + (1-p(k)) * V(b-1, k+1, 0)// miss -> pity advances
//
// BELLMAN OPTIMALITY. A finite-horizon MDP (every PULL decrements b, so at most
// b pulls can occur) admits an optimal DETERMINISTIC policy, so pi* collapses
// to one best action per state:
//
//   V*(s)  = max( q(s,SAVE), q(s,PULL) ) = max( 0, q(s,PULL) )
//   pi*(s) = argmax_a q(s,a)                          (ties -> SAVE)
//
// BOUNDARY:  V*(0, k, f) = 0          (no budget => no pull => G = 0)
//
// Because q(b, .,.) depends only on V*(b-1, .,.), we solve bottom-up in b
// (backward induction). Time and space O(B * N) (x2 for the flag f).
// ===========================================================================
struct Params 
{
    int    B    = 180;   // budget bound: max pulls the player may spend
    double cost = 1.0;   // c : cost paid per PULL (opportunity cost of currency)
    double W    = 94.0;  // W : reward (in pulls) for obtaining the featured unit
    double q    = 0.5;   // q : 50/50 win-rate, P(5-star is featured) when f = 0
};

// Per-state tables, indexed [b][k][f] with b=0..B, k=0..N-1, f in {0,1}.
template <class T>
using Grid = std::vector<std::vector<std::array<T, 2> > >;

struct Solution 
{
    Params prm;
    Grid<double> V;       // V[b][k][f] = optimal value function V*(b,k,f)
    Grid<Action> policy;  // policy[b][k][f] = optimal action pi*(s)
};

// Solve Objective B by backward induction on b. Globally optimal by Bellman
// optimality (see proof). Returns V* and the greedy-optimal policy pi*.
Solution solve_objectiveB(const Params& prm);

// Derived metrics evaluated UNDER pi* (not part of the return G, reported only
// for interpretation): probability of eventually obtaining the featured unit,
// and expected number of pulls actually spent.
struct PolicyEval 
{
    Grid<double> p_featured;  // P(get featured | start s, follow pi*)
    Grid<double> exp_pulls;   // E(pulls spent | start s, follow pi*)
};
PolicyEval evaluate_policy(const Solution& sol);

} // namespace gacha

#endif // GACHA_SOLVER_H
