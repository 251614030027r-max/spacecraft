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
| d̄_h | deviation threshold for hold | III-A |
| ρ_f, ρ_D | far-field range, descent range | III-A |
| T_ml, T_pr | mode-logic period, mode-prediction horizon | III-A |
| ρ_m, ρ_m(t) (bold) | mouth radius (also planning keep-out radius), mouth point | III-A |
| S, H, E, D (upright) | execution modes | III-A |
| x, 𝒳 | decision state, its space | III-A |
| P_E, r | macro-step kernel, macro-step reward | III-A |
| 𝓜, z | mode prediction, execution feedback | III-A |
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
| t_0, σ, 𝒟, f̄_0, f̄_1 | arrival instant, shift, shift set, entry thrust bounds | III-C |
| u, q, q_ref, N_c, W_q, W_u, W_N, A, B, w_0, ū, h_n, ε (bold), w_ε | MPC input, state, reference, horizon, weights, linear model, bounds, linearized margins, slack | III-D |
| t_k, k, j, 𝒥_k | decision instant/index, control-period index, periods of a macro step | IV-A |
| o, R_s, ι | observation, spin-basis matrix, mission indicator | IV-A |
| r_term, χ, G, γ | terminal reward, termination flag, return, per-decision discount | IV-A |
| r̃, Φ | shaped reward, potential | IV-A |
| T_o, d_ρ, d_v, i_H | feedback offset, previous-stage deviations, hold indicator | IV-B |
| π_θ, Q_ζi, ζ_i^- | policy, critics, target-critic parameters | IV-B |
| ε̂, τ̂, Δv̂ | predicted event indicator, event time, propellant | IV-C |
| 𝒢, 𝒢_a, 𝒞, ã, a* | candidate set, admissible set, correction map, executed intent, minimizer | IV-C |
| P_C, 𝓑^π, Π, V_C, V_E, Υ | screened kernel, soft Bellman operator, policy class, returns under P_C / P_E; Υ feedback law in Problem 1 (II-C) | IV-D |
| α, Q_tg, 𝓛, ℬ, U | SAC temperature, critic target, losses, replay buffer, soft value | IV-D |
| k_1, k_2, φ_r, c_w | auxiliary unit vectors, line-of-sight azimuth, cosine argument (proofs) | App |
