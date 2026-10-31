# Precapture v2 fast-track results: F1--F3, S3-prime, and S5

Date: 2026-09-06
Seed scope: direction-only three-seed checks; no formal 20-episode result

## F1: predictive MPC entry-plane activation -- PASS

The environment truth latch remains unchanged: only a legal outside-to-inside
entry-disc crossing latches the terminal region. MPC now activates terminal
corridor and speed rows independently at each nominal horizon state whose
predicted port-axial distance is inside the entry plane. The fixed QP dimension
is unchanged.

The three previously closing-speed-illegal h50 seeds all completed with zero
active-constraint violations and no illegal entry:

| Seed | Completion time (s) | Entry time (s) | Entry closing speed (m/s) | Entry total speed (m/s) | Delta-v (m/s) | Serial p95 (s) |
|---:|---:|---:|---:|---:|---:|---:|
| 262000 | 58.0 | 34.3 | 0.194475 | 0.223842 | 1.723182 | 0.130447 |
| 262002 | 62.1 | 36.9 | 0.194151 | 0.238764 | 1.724579 | 0.130053 |
| 262004 | 63.9 | 40.6 | 0.195367 | 0.262689 | 2.573685 | 0.149838 |

All entry closing speeds are at or below the 0.20 m/s entry limit. F1 restores
the missing look-ahead but does not make h50 real-time: all three serial p95
values exceed the 0.1 s control period.

## F2: retreat and legal re-entry semantics -- PASS

The deterministic state-sequence diagnostic first produces one illegal entry
outside the disc, then prescribes a retreat outside the entry plane, and finally
crosses inside the disc at legal speed. The terminal latch is false after the
illegal crossing and retreat, true after legal re-entry, and the illegal count
remains one. This proves that the geometry/state machine permits retreat and
re-entry. It is deliberately not presented as a controller demonstration.

## F3: fixed feasibility screen -- PARTIAL

The hand-guidance selector now considers local minima in time order and rejects
only candidates for which
`theta / (t_window - t_now) > 0.7 * sqrt((5/106) / radius)`. It does not rank
feasible windows. The public action remains a 3D waypoint with a 2 s hold.

Seeds 262001, 262002, and 262004 produced zero distance failures. Seeds 262002
and 262004 each skipped one infeasible first window and timed out safely; seed
262001 accepted its first window but terminated at 30.6 s on a small FOV
violation (-0.001269 rad). Completion was 0/3; failure split was distance 0,
timeout 2, FOV 1, corridor 0, speed 0. Thus the requested anti-divergence effect
is present, but the rule is not a usable completion baseline and was not tuned
further.

## S3-prime: Pure MPC direction check

| Controller | Complete | Truth-zero complete | Main failure | Completion time mean (s) | Completed delta-v mean (m/s) | Serial p95 (s) |
|---|---:|---:|---|---:|---:|---:|
| h20 | 2/3 | 2/3 | FOV 1 | 114.75 | 1.576261 | 0.054126 |
| h50 | 3/3 | 3/3 | none | 92.67 | 1.609139 | 0.135945 |

h50 now has the stronger direction-only completion result, while h20 is the
only one within the serial 0.1 s compute budget. These three-seed values are not
formal reporting data.

## S5: low-dimensional 3D waypoint interface -- STOP

The implemented family uses three target-centred, inertially oriented 3D
waypoint modes with two switch times: an outer co-rotating approach-axis gate,
an entry-axis gate, and the terminal waypoint. The lower controller accepts one
3D action every 2 s. A deterministic derivative-free candidate search uses the
declared lexicographic priority: truth safety/completion, completion time, then
delta-v. It does not replay the oracle trajectory.

The best valid common candidate in the checked family was 10 m outer gate, 6 m
entry gate, and switches at 20 s / 80 s. With a fixed 40 degree MPC planning FOV
against the unchanged real 50 degree geometry, it achieved 2/3 completion and
2/3 zero-violation completion. Its successful completion time was 159.05 s and
completed delta-v was 2.766416 m/s; serial p95 was 0.058467 s. Seed 262001 still
failed on FOV at 31.4 s. This does not improve on the current h20 direction row
(also 2/3, 114.75 s, 1.576261 m/s).

Tightening the planning FOV from 45 to 40 degrees was verified to reach the QP,
but it did not materially change the failing trajectory. As an independent
interface ceiling diagnostic, the pre-existing continuous oracle plan was sent
through the same h20, 3D/2 s channel on seed 262001; it also failed at 30.6 s on
FOV (-0.001269 rad). That oracle replay is diagnostic only and is not counted as
an S5 candidate.

Conclusion: the tested 3D waypoint interface plus current h20 lower loop did not
produce a clear improvement in success, time, or efficiency despite structured
search, and the failure is insensitive to the supplied translational waypoint.
The fast-track second stopping condition is therefore active. S6 training must
not start until the lower-loop/interface treatment of the seed-262001 FOV mode
is repaired and revalidated. The public action must remain 3D unless separately
authorized; the evidence points first to lower-loop attitude/FOV enforcement,
not to adding velocity or future-truth inputs.

## Evidence

- `logs/precapture_planning_v2/f1_h050_tw1000_seed262000.json`
- `logs/precapture_planning_v2/f1_h050_tw1000_seed262002.json`
- `logs/precapture_planning_v2/f1_h050_tw1000_seed262004.json`
- `logs/precapture_planning_v2/f2_scripted_reentry.json`
- `logs/precapture_planning_v2/f3_hand_guidance_seed262001.json`
- `logs/precapture_planning_v2/f3_hand_guidance_seed262002.json`
- `logs/precapture_planning_v2/f3_hand_guidance_seed262004.json`
- `logs/precapture_planning_v2/s3prime_mpc_h20_3seeds.json`
- `logs/precapture_planning_v2/s3prime_mpc_h50_3seeds.json`
- `logs/precapture_planning_v2/s5_piecewise_search_gate_stage1.json`
- `logs/precapture_planning_v2/s5_piecewise_candidate8_fov40.json`
- `logs/precapture_planning_v2/s5_interface_ceiling_oracle_seed262001.json`

The first six S5 candidates based on an initial-direction descent were rejected
as a flawed search family because they could cross the infinite entry plane
outside the disc before reaching the gate. Their untracked scratch output was
removed and is not evidence for the stopping decision.
