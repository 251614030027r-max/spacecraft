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
| s, Ω, T_s | spin axis, spin rate, spin period | II-A |
| p_port, n_p, p_cap, n | port, port axis, capture point (F_T), port direction in F_I | II-B |
| ℓ(·), ℓ_min, ϑ_c, K | axial coordinate, min axial distance, cone half-angle, approach cone | II-B |
| ρ_s, ℓ_m, K_m | operation-sphere radius, mouth band, mouth region | II-B |
| v_b(·), v_f | braking envelope, envelope floor speed | II-B |
| S, d_clr, ϑ_fov, v̄_m, ρ̄ | safe set, clearance, field-of-view half-angle, speed bound in K_m, range bound | II-B |
| X_cap, t_c, T_c | capture set, capture time, capture holding time | II-B |
| T_d, B̄, c_t, c_f | deadline, budget, time and propellant weights | II-C |
| t̄_e, T_r | latest admissible entry time, terminal reserve | II-C |
| τ_p, η | time to next pass center, opportunity margin | II-C |
| h, a=[a_p,a_z], A | stage duration, stage intent, action box | III-A |
| d̄_h, d̄_e | deviation thresholds (hold, entry re-plan) | III-A, III-C |
| ρ_f, ρ_D | far-field range, descent range | III-A |
| T_m, T_pr | mode-logic period, mode-prediction horizon | III-A |
| S, H, E, D (upright) | execution modes | III-A |
| x, 𝒳 | decision state, its space | III-A |
| P_E, r | macro-step kernel, macro-step reward | III-A |
| 𝓜, z | mode prediction, execution feedback | III-A |
| β, λ, δ, φ | sweep-cone half-angle, line-of-sight polar angle, latitude offset, phase | III-B |
| e_r, e_pol, e_az | spin-adapted basis | III-B |
| ψ, ψ̄, ψ̂, φ_w | port–line-of-sight angle, entry gate, predicted angle, window half-width | III-B |
| v*, v_cl, v_lat, v_sw, v_0, v_a | target velocity and its components, intent speed offset/scale | III-B |
| k_λ, v̄_lat, ρ̄_r | latitude gain, saturation, range bound in Prop. 1 | III-B |
| v_r, v_pol, v_az, κ | velocity components, phase-rate ratio | III-B |
| v_g, ρ_g, 𝒫, v̄_st, ρ_min | stage goal, safety projection, stage authority, min stage-end range | III-B |
| T, N, Δt_pl, f̄_pl, n | planning horizon, intervals, interval length, planning bound, interval index | III-C |
| L_f, w_ρ, w_g, w_v, 𝒳_f, ρ̄_pl | terminal cost, weights, terminal set, planning range bound | III-C |
| ρ_m, ρ_m(t) (bold), v_in | mouth radius/keep-out radius, mouth point, entry closing speed | III-C |
| 𝒯, τ_e, ρ_E, m_b, m_b,min | arrival times, entry arrival time, entry range, braking margin, its threshold | III-C |
| t_0, σ, Σ_0, Σ_1, f̄_0, f̄_1 | arrival instant, shift, shift sets, entry thrust bounds | III-C |
| N_mpc | MPC horizon (control periods) | III-D |
| t_k, k, j, 𝒥_k | decision instant/index, control-period index, periods of a macro step | IV-A |
| o, R_s, ι | observation, spin-basis matrix, mission indicator | IV-A |
| r_term, χ, G, γ | terminal reward, termination flag, return, per-decision discount | IV-A |
| r̃, Φ, ℓ̄_Φ, τ_Φ | shaped reward, potential, cap, look-ahead | IV-A |
| T_o, d_ρ, d_v, i_H | feedback offset, previous-stage deviations, hold indicator | IV-B |
| π_θ, Q_ζi, ζ_i^- | policy, critics, target-critic parameters | IV-B |
| ε̂, τ̂, Δv̂ | predicted event indicator, event time, propellant | IV-C |
| 𝒢, 𝒢_a, 𝒞, ã, a* | candidate set, admissible set, correction map, executed intent, minimizer | IV-C |
| P_C, B^π, Π, V_C, V_E | screened kernel, Bellman operator, policy class, returns under P_C / P_E | IV-D |
| α, y, 𝓛 | SAC temperature, critic target, losses | IV-D |
| k_1, k_2, φ_r, ς | auxiliary unit vectors, line-of-sight azimuth, cosine argument (proofs) | App |
