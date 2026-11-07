# P1-A: timing control and incremental hold-radius scan

Full-state / perception=None; unchanged MPC, dynamics, constraints, reward, 300 s budget and absolute waypoint parameterization. The only added policy freedom is the radius of the existing frozen inertial hold. No training or learned interface changes.

Timing-only control rescues [262005]; timing plus radius rescues [262005]. Adding hold radius beyond the fixed commit-time scan rescues **0 additional seed(s)**: [].
These are post-hoc per-seed action-space ceilings, **not proof of a learnable policy**. Results from different horizons are shown separately; rescue at some horizon does not imply rescue at every horizon.

Each cell below is `C/F; end time s; force impulse N s; worst normalized RK45 truth margin`. F means failure, so its time/fuel must not be read as a successful-mission performance figure. All predeclared cells are reported. Six final cells also reuse an earlier identical hold episode that terminated before its commit time (source references and reasons are in the manifest). Initial-radius cells reuse the identical timing-only control; at t=0 no hold command executes, so all radii reuse that same control trajectory.

## Seed 262005, h20

### Timing-only control

| Commit s | Result | Artifact |
|---|---|---|
| 0 | F; 96.1; 387.76; -0.002591 | `commit_s262005_h20_t0.json` |
| 30 | F; 109.6; 502.58; -0.0046494 | `commit_s262005_h20_t30.json` |
| 60 | C; 293.8; 354.83; 0.14264 | `commit_s262005_h20_t60.json` |
| 90 | C; 293.8; 354.83; 0.14264 | `commit_s262005_h20_t90.json` |
| 120 | C; 293.8; 354.83; 0.14264 | `commit_s262005_h20_t120.json` |
| 150 | C; 297.4; 403.52; 0.14201 | `commit_s262005_h20_t150.json` |

### Timing × radius increment

| Commit s | Initial radius | 12 m | 14 m | 16 m | 18 m |
|---|---|---|---|---|---|
| 0 | F; 96.1; 387.76; -0.002591 | F; 96.1; 387.76; -0.002591 | F; 96.1; 387.76; -0.002591 | F; 96.1; 387.76; -0.002591 | F; 96.1; 387.76; -0.002591 |
| 30 | F; 109.6; 502.58; -0.0046494 | F; 88.1; 400.82; -0.00075593 | F; 97.7; 427.99; -0.0086368 | F; 111.6; 517.01; -0.0066835 | F; 119.7; 559.63; -0.0066263 |
| 60 | C; 293.8; 354.83; 0.14264 | F; 63.6; 143.12; -0.0047286 | C; 271.2; 281.74; 0.14436 | C; 289.5; 332.30; 0.14262 | F; 300.0; 384.73; 0.14287 |
| 90 | C; 293.8; 354.83; 0.14264 | F; 63.6; 143.12; -0.0047286 | C; 271.2; 281.74; 0.14436 | C; 289.5; 332.30; 0.14262 | F; 300.0; 384.73; 0.14287 |
| 120 | C; 293.8; 354.83; 0.14264 | F; 63.6; 143.12; -0.0047286 | C; 271.2; 281.74; 0.14436 | C; 289.5; 332.30; 0.14262 | F; 300.0; 384.73; 0.14287 |
| 150 | C; 297.4; 403.52; 0.14201 | F; 63.6; 143.12; -0.0047286 | C; 273.3; 323.58; 0.14265 | C; 293.2; 379.95; 0.14201 | F; 300.0; 432.81; 0.14257 |

Radius-layer successful cells: 12/30. Only part of the fixed grid succeeds; the table exposes parameter sensitivity.

## Seed 262005, h35

### Timing-only control

| Commit s | Result | Artifact |
|---|---|---|
| 0 | F; 30.2; 179.28; -0.015334 | `commit_s262005_h35_t0.json` |
| 30 | F; 145.4; 802.70; -0.013229 | `commit_s262005_h35_t30.json` |
| 60 | C; 255.5; 443.83; 0.01947 | `commit_s262005_h35_t60.json` |
| 90 | C; 255.5; 443.83; 0.01947 | `commit_s262005_h35_t90.json` |
| 120 | C; 255.5; 443.83; 0.01947 | `commit_s262005_h35_t120.json` |
| 150 | C; 255.5; 443.83; 0.01947 | `commit_s262005_h35_t150.json` |

### Timing × radius increment

| Commit s | Initial radius | 12 m | 14 m | 16 m | 18 m |
|---|---|---|---|---|---|
| 0 | F; 30.2; 179.28; -0.015334 | F; 30.2; 179.28; -0.015334 | F; 30.2; 179.28; -0.015334 | F; 30.2; 179.28; -0.015334 | F; 30.2; 179.28; -0.015334 |
| 30 | F; 145.4; 802.70; -0.013229 | C; 91.1; 316.76; 0.019102 | F; 131.4; 799.30; -0.0050577 | F; 135.9; 769.71; -0.022154 | F; 88.2; 484.66; -0.0035596 |
| 60 | C; 255.5; 443.83; 0.01947 | C; 240.5; 373.43; 0.010054 | F; 149.0; 133.66; -2.6054e-05 | F; 169.9; 139.20; -0.00018521 | C; 257.5; 446.58; 0.019694 |
| 90 | C; 255.5; 443.83; 0.01947 | C; 240.5; 373.43; 0.010054 | F; 149.0; 133.66; -2.6054e-05 | F; 169.9; 139.20; -0.00018521 | C; 257.5; 446.58; 0.019694 |
| 120 | C; 255.5; 443.83; 0.01947 | C; 240.5; 373.43; 0.010054 | F; 149.0; 133.66; -2.6054e-05 | F; 169.9; 139.20; -0.00018521 | C; 257.5; 446.58; 0.019694 |
| 150 | C; 255.5; 443.83; 0.01947 | C; 236.8; 378.91; 0.010105 | F; 149.0; 133.66; -2.6054e-05 | F; 169.9; 139.20; -0.00018521 | C; 257.8; 466.94; 0.019654 |

Radius-layer successful cells: 13/30. Only part of the fixed grid succeeds; the table exposes parameter sensitivity.

## Seed 262006, h20

### Timing-only control

| Commit s | Result | Artifact |
|---|---|---|
| 0 | F; 35.0; 44.45; -0.0015316 | `commit_s262006_h20_t0.json` |
| 30 | F; 43.0; 36.65; -0.0011066 | `commit_s262006_h20_t30.json` |
| 60 | F; 43.0; 36.65; -0.0011066 | `commit_s262006_h20_t60.json` |
| 90 | F; 43.0; 36.65; -0.0011066 | `commit_s262006_h20_t90.json` |
| 120 | F; 43.0; 36.65; -0.0011066 | `commit_s262006_h20_t120.json` |
| 150 | F; 43.0; 36.65; -0.0011066 | `commit_s262006_h20_t150.json` |

### Timing × radius increment

| Commit s | Initial radius | 12 m | 14 m | 16 m | 18 m |
|---|---|---|---|---|---|
| 0 | F; 35.0; 44.45; -0.0015316 | F; 35.0; 44.45; -0.0015316 | F; 35.0; 44.45; -0.0015316 | F; 35.0; 44.45; -0.0015316 | F; 35.0; 44.45; -0.0015316 |
| 30 | F; 43.0; 36.65; -0.0011066 | F; 43.6; 50.88; -0.0015026 | F; 44.1; 47.83; -0.00041906 | F; 43.6; 40.22; -0.00018729 | F; 43.0; 37.23; -0.0021959 |
| 60 | F; 43.0; 36.65; -0.0011066 | F; 43.6; 50.88; -0.0015026 | F; 44.1; 47.83; -0.00041906 | F; 43.6; 40.22; -0.00018729 | F; 43.0; 37.23; -0.0021959 |
| 90 | F; 43.0; 36.65; -0.0011066 | F; 43.6; 50.88; -0.0015026 | F; 44.1; 47.83; -0.00041906 | F; 43.6; 40.22; -0.00018729 | F; 43.0; 37.23; -0.0021959 |
| 120 | F; 43.0; 36.65; -0.0011066 | F; 43.6; 50.88; -0.0015026 | F; 44.1; 47.83; -0.00041906 | F; 43.6; 40.22; -0.00018729 | F; 43.0; 37.23; -0.0021959 |
| 150 | F; 43.0; 36.65; -0.0011066 | F; 43.6; 50.88; -0.0015026 | F; 44.1; 47.83; -0.00041906 | F; 43.6; 40.22; -0.00018729 | F; 43.0; 37.23; -0.0021959 |

Radius-layer successful cells: 0/30. No cell succeeds.

## Seed 262006, h35

### Timing-only control

| Commit s | Result | Artifact |
|---|---|---|
| 0 | F; 29.8; 40.70; -0.0024862 | `commit_s262006_h35_t0.json` |
| 30 | F; 47.2; 42.13; -0.0010965 | `commit_s262006_h35_t30.json` |
| 60 | F; 47.2; 42.13; -0.0010965 | `commit_s262006_h35_t60.json` |
| 90 | F; 47.2; 42.13; -0.0010965 | `commit_s262006_h35_t90.json` |
| 120 | F; 47.2; 42.13; -0.0010965 | `commit_s262006_h35_t120.json` |
| 150 | F; 47.2; 42.13; -0.0010965 | `commit_s262006_h35_t120.json` |

### Timing × radius increment

| Commit s | Initial radius | 12 m | 14 m | 16 m | 18 m |
|---|---|---|---|---|---|
| 0 | F; 29.8; 40.70; -0.0024862 | F; 29.8; 40.70; -0.0024862 | F; 29.8; 40.70; -0.0024862 | F; 29.8; 40.70; -0.0024862 | F; 29.8; 40.70; -0.0024862 |
| 30 | F; 47.2; 42.13; -0.0010965 | F; 37.9; 40.45; -0.00010104 | F; 39.5; 39.12; -0.0011525 | F; 41.5; 36.46; -0.0011293 | F; 48.2; 42.74; -0.00077913 |
| 60 | F; 47.2; 42.13; -0.0010965 | F; 37.9; 40.45; -0.00010104 | F; 39.5; 39.12; -0.0011525 | F; 41.5; 36.46; -0.0011293 | F; 48.2; 42.74; -0.00077913 |
| 90 | F; 47.2; 42.13; -0.0010965 | F; 37.9; 40.45; -0.00010104 | F; 39.5; 39.12; -0.0011525 | F; 41.5; 36.46; -0.0011293 | F; 48.2; 42.74; -0.00077913 |
| 120 | F; 47.2; 42.13; -0.0010965 | F; 37.9; 40.45; -0.00010104 | F; 39.5; 39.12; -0.0011525 | F; 41.5; 36.46; -0.0011293 | F; 48.2; 42.74; -0.00077913 |
| 150 | F; 47.2; 42.13; -0.0010965 | F; 37.9; 40.45; -0.00010104 | F; 39.5; 39.12; -0.0011525 | F; 41.5; 36.46; -0.0011293 | F; 48.2; 42.74; -0.00077913 |

Radius-layer successful cells: 0/30. No cell succeeds.

## Gate and artifacts

Unrescued seeds: [262006]. Continue T3 on only these unrescued seeds.
Per-cell JSON and the exact grid/commands are in `logs/p1_a_20260910/`. Each JSON records actual impulse and worst truth margin over every micro step, not merely the final macro decision. The legacy qp_zero_fallbacks field still describes the final macro decision; qp_fallbacks_total is the episode total. No parallel time is used for compute claims.
