# T1B_V8_STAGE1_UPPER_REVIEW — attribution of the V8 stage-I closed loop (lower delivery 9c55ddc)

`stage1_attribution.py` reads `evidence/T1B_V8_STAGE1/` only; output `STAGE1_ATTRIBUTION.json`.

- 36 paired openings (292500-292503, 293100-293131); one rescue state among them (293117).
- Mean return minus the delivered no-verification E-SAC 10k: risk +5.32, random +5.57. Both arms
  verified 293117 at its first decision (random by its hash draw) and executed 140 s: +184.81 each.
  Without 293117: risk +0.19, random +0.45.
- Where the risk trigger spent its model branches (first-decision truth class of the opening):
  all-fail 40 / 54, rescue 5, earliest-best 6, better-alternative 3; triggered decisions 14 of 19
  in all-fail openings (re-triggered at each re-decision, every candidate predicted to fail, nothing
  changed). Random: 26 / 73 in all-fail, 18 in better-alternative openings, where its two other
  positive differences come from (293107 +4.12, 293125 +4.90).
- The critic's Q orders the candidates earliest > later in every triggered state, including the
  rescue 293117 (-12.4 / -22.2 / -32.7 vs truth -131.7 / -177.6 / +53.1)
  (`evidence/T1B_V8_STAGE1/STAGE1_RESULT.md`, triggered_openings).

Reading: the real-TD critic predicts that the default intent will fail (AUC 0.90), which is mostly
true in states where every intent fails; the value of verification is the gain of the best
alternative over the default, which the critic does not rank. A failure-risk trigger therefore
concentrates the MPC compute on hopeless states. With one rescue in 36 openings the stage-I
equal-compute test had no power either way (the registered G3 is NO-GO regardless).
