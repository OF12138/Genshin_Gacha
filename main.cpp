// Gacha optimal pull/save strategy -- OBJECTIVE B (probability of featured).
//
// Finite-horizon MDP solved exactly by backward induction (Bellman optimality).
// State s = (b, k, f): budget, pity counter, 50/50-guarantee flag.
// See solver.h for the full V_pi / q_pi formalism and pity.h for the cited rates.
//
// Outputs, for each (budget B, reward W) configuration:
//   * V*(s), P(featured), E(pulls) at the start state, printed to the console,
//   * the threshold curve k*(b) (smallest pity at which pi*(s) = PULL),
//   * a compact ASCII decision map of pi*,
//   * full CSVs (decision map of pi* + value function V*) for the Python plot.

#include "pity.h"
#include "solver.h"

#include <cstdio>
#include <string>
#include <vector>
#include <fstream>

using namespace gacha;

// Smallest pity k at which pi*(b,k,f) = PULL; -1 if SAVE for every k.
static int threshold(const Solution& sol, int b, int f) {
    for (int k = 0; k < N; ++k)
        if (sol.policy[b][k][f] == PULL) return k;
    return -1;
}

// Write pi* over the (b,k) grid for a fixed f: 1 = PULL, 0 = SAVE.
// Row b = 1..B, one line per b, N comma-separated values.
static void write_decision_csv(const Solution& sol, int f, const std::string& path) {
    std::ofstream out(path.c_str());
    for (int b = 1; b <= sol.prm.B; ++b) {
        for (int k = 0; k < N; ++k)
            out << (sol.policy[b][k][f] == PULL ? 1 : 0) << (k + 1 < N ? "," : "\n");
    }
}

// Write the value function V*(b,k,f) over the (b,k) grid for a fixed f.
static void write_value_csv(const Solution& sol, int f, const std::string& path) {
    std::ofstream out(path.c_str());
    for (int b = 1; b <= sol.prm.B; ++b) {
        for (int k = 0; k < N; ++k)
            out << sol.V[b][k][f] << (k + 1 < N ? "," : "\n");
    }
}

// Compact ASCII map of pi* for a fixed f: 'P' = PULL, '.' = SAVE.
// Rows sampled every bstep, columns every kstep, to fit the console.
static void print_ascii_map(const Solution& sol, int f, int bstep, int kstep) {
    std::printf("    pity k -> 0%*s%d\n", (N / kstep) - 3, "", N - 1);
    for (int b = sol.prm.B; b >= 1; b -= bstep) {
        std::printf("b=%4d | ", b);
        for (int k = 0; k < N; k += kstep)
            std::putchar(sol.policy[b][k][f] == PULL ? 'P' : '.');
        std::putchar('\n');
    }
}

static std::string wtag(double W) {
    // 94 -> "94", 62.5 -> "62.5"
    if (W == (long long)W) return std::to_string((long long)W);
    std::string s = std::to_string(W);
    while (!s.empty() && s.back() == '0') s.pop_back();
    if (!s.empty() && s.back() == '.') s.pop_back();
    return s;
}

static void run(int B, double W, bool verbose) {
    Params prm;
    prm.B = B;
    prm.W = W;
    Solution sol = solve_objectiveB(prm);
    PolicyEval ev = evaluate_policy(sol);

    const std::string tag = "B" + std::to_string(B) + "_W" + wtag(W);
    write_decision_csv(sol, 0, "decision_f0_" + tag + ".csv");
    write_decision_csv(sol, 1, "decision_f1_" + tag + ".csv");
    write_value_csv(sol, 0, "value_f0_" + tag + ".csv");

    std::printf("==================================================================\n");
    std::printf(" OBJECTIVE B | B=%d  W=%.4g pulls  c=%.2g  q=%.2g  (hard pity N=%d)\n",
                B, W, prm.cost, prm.q, N);
    std::printf("------------------------------------------------------------------\n");
    // Headline: a fresh player at the start of the banner = (b=B, k=0, f=0).
    std::printf(" Start state s=(b=%d, k=0, f=0):\n", B);
    std::printf("   optimal action pi*(s) : %s\n",
                sol.policy[B][0][0] == PULL ? "PULL" : "SAVE");
    std::printf("   value  V*(s)          : %+.3f  (in pulls)\n", sol.V[B][0][0]);
    std::printf("   P(get featured)       : %.4f\n", ev.p_featured[B][0][0]);
    std::printf("   E(pulls spent)        : %.2f\n", ev.exp_pulls[B][0][0]);

    if (verbose) {
        // Threshold curve k*(b) for f=0 (no guarantee) and f=1 (guaranteed).
        std::printf("\n Threshold curve k*(b) = smallest pity k with pi*(b,k,f)=PULL\n");
        std::printf("   (-1 = never pull at this budget; 0 = always pull)\n");
        std::printf("   %6s | %10s | %10s\n", "budget", "f=0", "f=1");
        int step = (B > 30) ? B / 18 : 1;
        for (int b = B; b >= 1; b -= step) {
            int t0 = threshold(sol, b, 0), t1 = threshold(sol, b, 1);
            std::printf("   %6d | %10d | %10d\n", b, t0, t1);
        }

        std::printf("\n Decision map pi* (f=0, no guarantee).  P = PULL, . = SAVE.\n");
        print_ascii_map(sol, 0, (B > 40 ? B / 25 : 1), 1);
    }
    std::printf("\n CSVs written: decision_f{0,1}_%s.csv, value_f0_%s.csv\n\n",
                tag.c_str(), tag.c_str());
}

int main() 
{
    std::printf("\nGacha pull/save -- Objective B (maximize E[W*1{featured} - c*pulls])\n");
    std::printf("Genshin 5-star pity curve; classic 50/50 (q=0.5); c=1 pull/pull.\n\n");

    // Headline configurations: featured worth W=94 pulls, both budgets.
    run(180, 94.0, /*verbose=*/true);
    run(90,  94.0, /*verbose=*/true);

    // W-sweep (quiet): how the strategy shifts with the prize's worth.
    std::printf("################  W sweep (B=180)  ################\n");
    run(180, 62.5, /*verbose=*/false);
    run(180, 180.0, /*verbose=*/false);

    return 0;
}
