# Correction to review_readout.json (2026-10-13)

`review_readout.json` (produced by `experiments/t1_v2_review.py` at 818fcd8, reproducible
there) computed the planned arrival as `2 k + tau_effective_s` on every decision. G*'s
`tau_effective_s` is the total duration of the live plan from its start, so on a kept
decision that value drifts by +2 s although the plan and its arrival are unchanged. It also
mis-classified some new plans as later/earlier.

`review_readout_corrected.json` uses plan start + tau (the script after 818fcd8):

- 30k new plans: later 333, earlier 156 (was 277 / 212); 25k and other marks likewise changed.
- Failing development episodes: the final planned arrival at 30k is 351-376 s; at 25k
  three of the five timeouts end with a planned arrival of 281-296 s (not beyond 300 s)
  and the unsafe episode at 279 s.
  The statement "all 22 timeouts end with a planned arrival >= 298 s" (collab review
  24e2a44 / brief e7678b9) is withdrawn.
- Unchanged: in every failing episode the planned arrival moves later over the episode
  (from a minimum of 60-142 s to 279-519 s).

The original file is kept unchanged (it is cited by SHA in the registry).
