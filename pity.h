#ifndef GACHA_PITY_H
#define GACHA_PITY_H

namespace gacha 
{

// Hard-pity cap. The pity counter k runs 0..N-1.
// A pull made at counter k is the (k+1)-th pull since the last 5-star.
// Obtaining a 5-star resets k to 0.
constexpr int N = 90;

// p(k): probability that a single pull made at pity counter k yields a 5-star.
//
// SOURCE: Genshin Impact 5-star character-banner curve, reconstructed from
// large community statistical studies (millions of recorded wishes; summarized
// e.g. at dignitas.gg "Taking Advantage of the Pity System" and
// videogamer.com's pity explainer). The model:
//     k =  0..72  : base rate 0.6%
//     k = 73..88  : "soft pity" linear ramp, p(k) = 0.006 + 0.06*(k-72)
//                   (the 74th pull, k=73, is ~6.6%)
//     k = 89      : "hard pity" -> guaranteed (p = 1.0) on the 90th pull
// The effective rate averages ~1.6%/pull => one 5-star per ~62.5 pulls.
// This function is intentionally the ONLY place rates live, so the model is
// swappable to another game.
double pity_prob(int k);

} // namespace gacha

#endif // GACHA_PITY_H
