# Symbol table (self-check, not part of the paper)

One symbol, one meaning; one quantity, one symbol. Conventions: bold = vector/matrix; hat = predicted;
bar = upper bound; tilde = transformed; Δ = increment only; upright subscripts are names.
"First" = section of first appearance (II-A = Section II-A, App = appendix, V-A = setup).

| Symbol | Meaning | First |
|---|---|---|
| F_I, F_T, F_C | inertial, target-body, chaser-body frames | II-A |
| e_C | chaser boresight unit vector (body x-axis) | II-A |
| g=(R,p), ξ=[ω;ν] | pose on SE(3), body twist (angular, translational velocity) | II-A |
| m, J, m_C | mass, inertia matrix, chaser mass | II-A |
| f, Γ | control force, control torque (body) | II-A |
| f_g, Γ_g, M, μ, p, f_{J2} | gravity force/torque, second-moment matrix, gravitational parameter, orbital radius, J2 force | II-A |
| g_r, ξ_r | relative pose and twist | II-A |
| ρ (bold), ρ, v, ρ_T | relative position, range, relative velocity, position in F_T | II-A |
| X | joint state of the two spacecraft | II-A |
| Δt | control period | II-A |
| f̄, Γ̄ | per-axis force / torque bounds (∞-norm) | II-A |
| Δv(t) | velocity increment (propellant) used up to t | II-A |
| R̂_T(t+τ) | constant-rate attitude prediction | II-A |
| s, Ω, T_sp | spin axis, spin rate, spin period | II-A |
| p_port, n_p, p_cap, n | port, port axis, capture point (F_T), port direction in F_I | II-B |
| ℓ(·), ℓ_min, ϑ_c, K | axial coordinate, min axial distance, cone half-angle, approach cone | II-B |
| ρ_s, ℓ_m, K_m | operation-sphere radius, mouth band, mouth region | II-B |
| v_b(d), v_f, v_mg | braking envelope of distance d, envelope floor speed, radial braking margin | II-B |
| S, d_clr, ϑ_fov, v̄_m, ρ̄ | safe set, clearance, field-of-view half-angle, speed bound in K_m, range bound | II-B |
| S_cap, t_c, T_hold | capture set, capture time (∞ if not captured), holding time | II-B |
| T_d, B̄, c_t, c_f | deadline, budget, time and propellant weights | II-C |
| t̄_e, T_r | latest admissible entry time, terminal reserve | II-C |
| τ_p, η | time to next pass center, opportunity margin | II-C |
| h, a=[a_p,a_z], A | stage duration, stage intent, action box | III-A |
| T_ml, T_pr | mode-logic period, mode-prediction horizon | III-A |
| ρ_m, ρ_m(t) (bold) | mouth radius (also planning keep-out radius), mouth point | III-A |
| S, H, E, D (upright) | execution modes | III-A |
| x, 𝒳 | decision state, its space | III-A |
| P_E, r | macro-step kernel, macro-step reward | III-A |
| β, λ, δ, φ | sweep-cone half-angle, line-of-sight polar angle, latitude offset, phase | III-B |
| e_r, e_pol, e_az | spin-adapted basis | III-B |
| ψ, ψ̂(τ), ψ̄, φ_w | current and predicted port–line-of-sight angle, entry gate, window half-width | III-B |
| v*, v_cl, v_lat, v_sw, v_0, v_a | target velocity and its components, intent speed offset/scale | III-B |
| k_λ, v̄_lat | latitude gain, saturation | III-B |
| v_r, v_pol, v_az, κ | velocity components, phase-rate ratio | III-B |
| v_g, ρ_g, 𝒫, v̄_st, v_mg,min | stage goal, safety projection, stage authority, margin threshold | III-B |
| T, N, Δt_pl, f̄_pl, n, f_I | planning horizon, intervals, interval length, planning bound, interval/step index, inertial-axes force | II-A, III-C |
| L_f, w | terminal cost, propellant weight in Problem 2 | III-C |
| v_in, n̂ | entry closing speed, predicted port direction | III-C |
| 𝒯, τ_e, ρ_E | arrival times, entry arrival time, entry range | III-C |
| f̄_0 | entry thrust bound (re-timing uses a larger bound, named in words) | III-C |
| u, q, q̃_n, q_ref, N_c, W, w_u, w_N, A, B, d_0, ū, h_n, ε (bold), w_ε, ε̄ | MPC input, state, tracking error, reference, horizon, state-scale weight, input and terminal weights, linear model and offset, bounds, linearized margins, slack, slack weight and bound | III-D |
| t_k, k, j, 𝒥_k | decision instant/index, control-period index, periods of a macro step | IV-A |
| o, R_s | observation (17 entries, policy and critics), spin-basis matrix | IV-A |
| r_term, χ, G, γ | terminal reward, termination flag, return, per-decision discount | IV-A |
| r̃, Φ, T_go, Φ̄, τ_m, κ̄, |·|_2π | shaped reward, opportunity-reachability potential (weight c_t), reachability time, cap, nominal closing time, phase-rate bound, wrapped angle | IV-A |
| π_θ, Q_ζi, ζ_i^- | policy, critics, target-critic parameters | IV-C |
| 𝓜, ε̂, τ̂, Δv̂ | mode prediction, predicted event indicator, event time, propellant | IV-B |
| 𝒢, 𝒢_c, 𝒞, a* | candidate set, critical candidate set, coordination map, nearest critical candidate (corrected intent) | IV-B |
| 𝒯_L, τ_L, launch(τ_L), 𝒳_lp | long-range arrival times, long-range arrival time, launch command, last-usable-pass states | IV-B |
| P_C, 𝓑^π, Q^π_C, Υ | coordinated kernel, soft Bellman operator, its fixed point; Υ feedback law in Problem 1 (II-C) | IV-C |
| α, Q_tg, 𝓛, ℬ, U | SAC temperature, critic target, losses, replay buffer, soft value | IV-C |
| e_p, e_v, e_att, e_ω | position, velocity, attitude, angular-velocity errors w.r.t. the capture pose | V-C |
| m_cor, m_ax, m_v, ϑ_v | corridor, axial-braking, relative-speed margins; boresight–port view angle (Fig. 7) | V-C |
| k_1, k_2, φ_r, c_w | auxiliary unit vectors, line-of-sight azimuth, cosine argument (proofs) | App |
