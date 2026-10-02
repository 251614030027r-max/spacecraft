"""The precapture attitude reference, and what each mode asks the chaser to do.

Reference positions for this task are written in the *target* body frame, so a
waypoint held fixed in inertial space sweeps through that frame at the target's
tumble rate. ``frozen`` -- the default, and what every recorded precapture row
was measured under -- answers that sweep with one held attitude and a zero
relative-rate reference. These pin what the two diagnostic alternatives change
and, more importantly, what they must not change.
"""

from __future__ import annotations

import numpy as np

from controllers.mpc import MPCController
from controllers.mpc.prediction import LocalRelativePredictionModel
from env.scenarios import chaser_parameters


def _precapture_reference_for_mode(mode: str, *, reference_source: str):
    from controllers.mpc import precapture_mpc_config
    from env.scenarios import target_initial_state

    config = precapture_mpc_config(
        horizon_steps=4,
        reference_source=reference_source,
        precapture_attitude_reference=mode,
    )
    controller = MPCController(
        config, LocalRelativePredictionModel(chaser_parameters())
    )
    target = target_initial_state(tumble_scale=0.20)
    state = np.zeros(12)
    state[3:6] = np.array([-12.0, 4.0, 1.0])
    waypoint = target.rotation @ np.array([-11.0, 2.0, 0.5])
    return config, target, controller._reference_trajectory(
        state,
        target_state=target,
        external_reference=waypoint,
        terminal_latched=False,
    )


def test_precapture_attitude_reference_defaults_to_the_measured_frozen_mode() -> None:
    """The default is the mode every recorded precapture row was measured under.

    ``swept`` and ``aimed`` were added as diagnostics; making either of them the
    default would silently move F1, S3' and every earlier MPC number.
    """

    from controllers.mpc import precapture_mpc_config

    assert precapture_mpc_config().precapture_attitude_reference == "frozen"


def test_precapture_attitude_modes_agree_when_the_reference_does_not_move() -> None:
    """A target-frame-stationary reference has no sightline sweep to follow.

    ``frozen`` and ``swept`` are then bitwise identical, which is what keeps
    the ``fixed`` reference row -- and every number measured on it -- unmoved
    by this option existing. ``aimed`` still differs, and only in the way it is
    defined to: it aims from the *reference* position rather than the state's,
    so it carries the parallax between the two as a constant pose offset with
    the rate rows still zero.
    """

    _, _, frozen = _precapture_reference_for_mode("frozen", reference_source="fixed")
    _, _, swept = _precapture_reference_for_mode("swept", reference_source="fixed")
    _, _, aimed = _precapture_reference_for_mode("aimed", reference_source="fixed")
    assert np.array_equal(frozen, swept)
    assert np.array_equal(frozen[6:9, :], np.zeros_like(frozen[6:9, :]))
    assert np.array_equal(aimed[6:9, :], np.zeros_like(aimed[6:9, :]))
    # Rows 3:6 are the exponential translation rho = J_l(phi)^-1 p, not the
    # position, so a different attitude moves them even at the same point;
    # compare the poses themselves.
    from dynamics.lie import se3_exp

    for index in range(frozen.shape[1]):
        assert np.allclose(
            se3_exp(aimed[:6, index])[:3, 3], se3_exp(frozen[:6, index])[:3, 3]
        )
    assert not np.allclose(aimed[:3, :], frozen[:3, :])


def test_precapture_swept_reference_rate_matches_the_target_tumble_rate() -> None:
    """An inertially held waypoint sweeps the target frame at ``omega``.

    ``frozen`` answers that sweep with a zero relative-rate reference, so the
    pose and rate halves of the reference describe different motions. The two
    new modes ask for the sweep itself, which is the target's own tumble rate.
    """

    config, target, frozen = _precapture_reference_for_mode(
        "frozen", reference_source="external_local"
    )
    tumble_rate = float(np.linalg.norm(target.omega))
    assert np.allclose(frozen[6:9, :], 0.0)
    for mode in ("swept", "aimed"):
        _, _, reference = _precapture_reference_for_mode(
            mode, reference_source="external_local"
        )
        rates = np.linalg.norm(reference[6:9, :], axis=0)
        assert np.all(rates > 0.5 * tumble_rate)
        assert np.all(rates < 2.0 * tumble_rate)
        # No parallax step in the first rate row -- see _precapture_reference_rotations.
        assert abs(float(rates[0] - rates[1])) < 0.1 * tumble_rate


def test_target_frame_channel_reproduces_the_direct_reference() -> None:
    """The 3D waypoint channel is lossless only in the target body frame.

    Handing the channel the same point the ``fixed`` row tracks must produce
    the same reference, otherwise no upper layer can express through it even
    the guidance the lower layer already flies. Under the ``inertial``
    contract it does not: a body-fixed goal has to be re-issued as a rotating
    inertial point, which the horizon map turns back into a circular reference.
    """

    from controllers.mpc import precapture_mpc_config
    from env.scenarios import target_initial_state

    task = precapture_mpc_config().precapture_task
    target = target_initial_state(tumble_scale=0.20)
    state = np.zeros(12)
    state[3:6] = np.array([-9.0, 3.0, 1.0])

    def reference_for(source: str, frame: str, waypoint):
        config = precapture_mpc_config(
            horizon_steps=6,
            reference_source=source,
            external_reference_frame=frame,
        )
        controller = MPCController(
            config, LocalRelativePredictionModel(chaser_parameters())
        )
        return controller._reference_trajectory(
            state,
            target_state=target,
            external_reference=waypoint,
            terminal_latched=False,
        )

    direct = reference_for("fixed", "inertial", None)
    through_target_frame = reference_for(
        "external_local", "target", task.desired_position
    )
    assert np.allclose(direct, through_target_frame)

    through_inertial = reference_for(
        "external_local", "inertial", target.rotation @ task.desired_position
    )
    assert not np.allclose(direct, through_inertial)
    # The inertial contract turns the stationary goal into a moving one.
    assert np.linalg.norm(through_inertial[9:12, -1]) > 0.05
    assert np.linalg.norm(direct[9:12, -1]) < 1.0e-12


def test_external_reference_frame_defaults_to_the_measured_contract() -> None:
    from controllers.mpc import precapture_mpc_config

    assert precapture_mpc_config().external_reference_frame == "inertial"
