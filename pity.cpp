#include "pity.h"

namespace gacha 
{

double pity_prob(int k) 
{
    if (k < 0) k = 0;
    if (k >= N - 1) return 1.0;          // k = 89 -> 90th pull is guaranteed
    if (k < 73) return 0.006;            // base rate 0.6%
    double p = 0.006 + 0.06 * (k - 72);  // soft-pity linear ramp (k=73 -> 6.6%)
    return p > 1.0 ? 1.0 : p;
}

} // namespace gacha
