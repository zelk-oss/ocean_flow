# -*- coding: utf-8 -*-
r'''Tests for src/forecast/forecast_model.py.

ForecastModel: state management, stepping, and advance.
'''

# System modules
from typing import Type
from unittest.mock import MagicMock, patch

# External modules
import lightning.fabric
import numpy as np
import pytest
import torch

# Internal modules
from {{cookiecutter.project_slug}}.modules.forecast_module import ForecastModule
from {{cookiecutter.project_slug}}.forecast.forecast_model import ForecastModel
from {{cookiecutter.project_slug}}.pipelines import PrePipeline, PostPipeline
from tests.conftest import (
    IdentityPreModule,
    IdentityPostModule,
    DummyNetwork,
    make_fabric,
)


# -----------------------------------------------------------
# Concrete ForecastModule subclasses for testing
# -----------------------------------------------------------

class SimpleModule(ForecastModule):
    r'''Doubles the last time-step of each state variable.

    Minimal ForecastModule used as a deterministic test
    surrogate. For an all-ones input the output sequence is
    2, 4, 8, ...
    '''

    def forward(
            self,
            states_surface: torch.Tensor,
            states_levels: torch.Tensor,
            **kwargs: object,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r'''Return doubled last time-step of each input.'''
        return (
            states_surface[:, -1:] * 2,
            states_levels[:, -1:] * 2,
        )


class ForcingRecorderModule(ForecastModule):
    r'''Records the raw forcing tensor at each step.

    Attributes
    ----------
    recorded_forcing : list
        List of tensors recorded during forward calls.
    '''

    def __init__(
            self, *args: object, **kwargs: object,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.recorded_forcing: list = []

    def forward(
            self,
            states_surface: torch.Tensor,
            states_levels: torch.Tensor,
            forcing_var: torch.Tensor = None,
            **kwargs: object,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r'''Record forcing_var and pass states through.'''
        self.recorded_forcing.append(
            forcing_var.clone()
            if forcing_var is not None
            else None
        )
        return (
            states_surface[:, -1:],
            states_levels[:, -1:],
        )


class ForcingConsumerModule(ForecastModule):
    r'''Indexes forcing along the time dim.

    Raises ``IndexError`` when the time dimension is empty,
    documenting the implicit contract that forcings must have
    at least one time step per call.
    '''

    def forward(
            self,
            states_surface: torch.Tensor,
            states_levels: torch.Tensor,
            forcing_var: torch.Tensor = None,
            **kwargs: object,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r'''Index element 0 along time; raise if empty.'''
        _ = forcing_var[:, 0]
        return (
            states_surface[:, -1:],
            states_levels[:, -1:],
        )


# -----------------------------------------------------------
# Helper functions
# -----------------------------------------------------------


def _build_model(
        module: ForecastModule,
) -> ForecastModel:
    r'''Wrap module in a CPU/float32 ForecastModel.

    Parameters
    ----------
    module : ForecastModule
        Module to wrap.

    Returns
    -------
    ForecastModel
        Configured model on CPU.
    '''
    return ForecastModel(
        module=module,
        fabric=make_fabric(),
        dtype=torch.float32,
        compile=False,
    )


def _build_model_nsteps(
        n_in: int,
        n_out: int,
) -> ForecastModel:
    r'''Wrap SimpleModule with explicit step parameters.

    Parameters
    ----------
    n_in : int
        Number of input time steps (n_in_steps).
    n_out : int
        Number of output time steps per call (n_out_steps).

    Returns
    -------
    ForecastModel
        Configured model on CPU with given step parameters.
    '''
    return ForecastModel(
        module=_make_module(cls=SimpleModule),
        fabric=make_fabric(),
        dtype=torch.float32,
        compile=False,
        n_in_steps=n_in,
        n_out_steps=n_out,
    )


def _make_module(
        cls: Type[ForecastModule] = SimpleModule,
) -> ForecastModule:
    r'''Construct a ForecastModule with identity pipelines.

    Parameters
    ----------
    cls : Type[ForecastModule], optional
        Subclass to instantiate. Default is SimpleModule.

    Returns
    -------
    ForecastModule
        Instantiated module.
    '''
    return cls(
        network=DummyNetwork(),
        pre_pipeline=PrePipeline(
            states_surface=IdentityPreModule(),
            states_levels=IdentityPreModule(),
        ),
        post_pipeline=PostPipeline(
            states_surface=IdentityPostModule(),
            states_levels=IdentityPostModule(),
        ),
    )


def _ones_state(batch: int = 1) -> dict:
    r'''Return an all-ones initial state dict.

    Parameters
    ----------
    batch : int, optional
        Batch size. Default is 1.

    Returns
    -------
    dict
        State dict with ``states_surface`` and
        ``states_levels`` filled with ones.
    '''
    return {
        "states_surface": np.ones(
            (batch, 1, 2, 4, 8), dtype=np.float32,
        ),
        "states_levels": np.ones(
            (batch, 1, 2, 3, 4, 8), dtype=np.float32,
        ),
    }


def _ones_state_n(
        n_time: int,
        batch: int = 1,
) -> dict:
    r'''Return an all-ones state dict with n_time steps.

    Parameters
    ----------
    n_time : int
        Number of time steps in the state window.
    batch : int, optional
        Batch size. Default is 1.

    Returns
    -------
    dict
        State dict with ``states_surface`` shape
        ``(batch, n_time, 2, 4, 8)`` and
        ``states_levels`` shape
        ``(batch, n_time, 2, 3, 4, 8)`` filled with ones.
    '''
    return {
        "states_surface": np.ones(
            (batch, n_time, 2, 4, 8),
            dtype=np.float32,
        ),
        "states_levels": np.ones(
            (batch, n_time, 2, 3, 4, 8),
            dtype=np.float32,
        ),
    }


def _surface_state(
        batch: int = 2,
        n_time: int = 1,
        n_var: int = 2,
        n_lat: int = 4,
        n_lon: int = 8,
) -> dict:
    r'''Create a random surface state dict.

    Parameters
    ----------
    batch : int, optional
        Batch size. Default is 2.
    n_time : int, optional
        Number of time steps. Default is 1.
    n_var : int, optional
        Number of surface variables. Default is 2.
    n_lat : int, optional
        Number of latitude points. Default is 4.
    n_lon : int, optional
        Number of longitude points. Default is 8.

    Returns
    -------
    dict
        State dict with random ``states_surface`` and
        ``states_levels``.
    '''
    rng = np.random.default_rng(seed=20260225)
    return {
        "states_surface": rng.standard_normal(
            (batch, n_time, n_var, n_lat, n_lon),
        ).astype(np.float32),
        "states_levels": rng.standard_normal(
            (batch, n_time, 2, 3, n_lat, n_lon),
        ).astype(np.float32),
    }


def _build_recorder_model_n1() -> ForecastModel:
    r'''Build ForcingRecorderModule model, n_in=n_out=1.

    Constructs a ForecastModel wrapping
    ForcingRecorderModule with n_in_steps=1 and
    n_out_steps=1.  State is initialised to all-ones with
    batch=1, n_time=1.

    Returns
    -------
    ForecastModel
        Model ready for recording forcing windows.
    '''
    mod = _make_module(cls=ForcingRecorderModule)
    m = ForecastModel(
        module=mod,
        fabric=make_fabric(),
        dtype=torch.float32,
        compile=False,
        n_in_steps=1,
        n_out_steps=1,
    )
    m.set_state(_ones_state_n(n_time=1, batch=1))
    return m


# -----------------------------------------------------------
# Fixtures
# -----------------------------------------------------------

@pytest.fixture()
def simple_module() -> SimpleModule:
    r'''Return a SimpleModule instance.'''
    return _make_module(cls=SimpleModule)


@pytest.fixture()
def model(simple_module: SimpleModule) -> ForecastModel:
    r'''ForecastModel wrapping SimpleModule on CPU/float32.'''
    return _build_model(simple_module)


@pytest.fixture()
def recorder_model() -> ForecastModel:
    r'''ForecastModel wrapping ForcingRecorderModule.

    State is initialised to all-ones with batch=1.
    '''
    mod = _make_module(cls=ForcingRecorderModule)
    m = _build_model(mod)
    m.set_state(_ones_state(batch=1))
    return m


@pytest.fixture()
def simple_model_ones() -> ForecastModel:
    r'''ForecastModel wrapping SimpleModule with ones state.

    State is initialised to all-ones with batch=1.
    '''
    mod = _make_module(cls=SimpleModule)
    m = _build_model(mod)
    m.set_state(_ones_state(batch=1))
    return m


# -----------------------------------------------------------
# Functional tests
# -----------------------------------------------------------

class TestForecastModelFunctional:
    r'''End-to-end functional tests for ForecastModel.'''

    def test_set_and_get_state_roundtrip(
            self, model: ForecastModel,
    ) -> None:
        r'''Values via set_state returned by get_state.'''
        arr = np.ones((2, 3), dtype=np.float32)
        model.set_state({"x": arr})
        result = model.get_state()
        np.testing.assert_array_equal(result["x"], arr)

    def test_advance_returns_correct_shape(
            self, model: ForecastModel,
    ) -> None:
        r'''advance(3) returns shape (batch, 3, ...).'''
        state = _surface_state(batch=2)
        model.set_state(state)
        preds = model.advance(n=3)
        assert preds["states_surface"].shape[1] == 3
        assert preds["states_surface"].shape[0] == 2

    def test_advance_one_step_matches_step(
            self, model: ForecastModel,
    ) -> None:
        r'''advance(1) output matches a single _step.

        With torch.cat, advance(1) trajectory has shape
        (B, 1, vars, lat, lon).  Element [:, 0] equals
        the last time-step of the manual _step state,
        i.e. pred_manual["states_surface"][:, -1].
        '''
        state = _surface_state(batch=2)
        model.set_state(state)
        pred_advance = model.advance(n=1)

        model.set_state(state)
        model._autoregressive_step()
        pred_manual = model.get_state()

        np.testing.assert_allclose(
            pred_advance["states_surface"][:, 0],
            pred_manual["states_surface"][:, -1],
            rtol=1e-5,
        )

    def test_advance_with_forcings(
            self, model: ForecastModel,
    ) -> None:
        r'''advance with per-step forcings succeeds.'''
        batch = 2
        n_steps = 4
        state = _surface_state(batch=batch)
        model.set_state(state)
        forcings = {
            "forcing_var": np.ones(
                (batch, n_steps + 1, 4, 8),
                dtype=np.float32,
            ),
        }
        preds = model.advance(
            n=n_steps, forcings=forcings,
        )
        assert (
            preds["states_surface"].shape[1] == n_steps
        )

    def test_advance_without_forcings(
            self, model: ForecastModel,
    ) -> None:
        r'''advance without forcings returns predictions.'''
        state = _surface_state(batch=2)
        model.set_state(state)
        preds = model.advance(n=2)
        assert "states_surface" in preds
        assert "states_levels" in preds

    def test_advance_n3_step0_value_is_2(
            self, simple_model_ones: ForecastModel,
    ) -> None:
        r'''First step has value 1.0 * 2^1 = 2.0.'''
        preds = simple_model_ones.advance(n=3)
        np.testing.assert_allclose(
            preds["states_surface"][:, 0], 2.0, rtol=0,
        )

    def test_advance_n3_step1_value_is_4(
            self, simple_model_ones: ForecastModel,
    ) -> None:
        r'''Second step has value 1.0 * 2^2 = 4.0.'''
        preds = simple_model_ones.advance(n=3)
        np.testing.assert_allclose(
            preds["states_surface"][:, 1], 4.0, rtol=0,
        )

    def test_advance_n3_step2_value_is_8(
            self, simple_model_ones: ForecastModel,
    ) -> None:
        r'''Third step has value 1.0 * 2^3 = 8.0.'''
        preds = simple_model_ones.advance(n=3)
        np.testing.assert_allclose(
            preds["states_surface"][:, 2], 8.0, rtol=0,
        )

    def test_advance_n3_levels_step0_value_is_2(
            self, simple_model_ones: ForecastModel,
    ) -> None:
        r'''states_levels first step has value 2.0.'''
        preds = simple_model_ones.advance(n=3)
        np.testing.assert_allclose(
            preds["states_levels"][:, 0], 2.0, rtol=0,
        )

    def test_advance_n3_levels_step2_value_is_8(
            self, simple_model_ones: ForecastModel,
    ) -> None:
        r'''states_levels third step has value 8.0.'''
        preds = simple_model_ones.advance(n=3)
        np.testing.assert_allclose(
            preds["states_levels"][:, 2], 8.0, rtol=0,
        )

    def test_state_after_advance_equals_final_step(
            self, simple_model_ones: ForecastModel,
    ) -> None:
        r'''Persisted state equals the last prediction.

        With torch.cat trajectories have shape
        (B, n_calls*n_out_steps, ...).  The internal state
        has shape (B, n_in_steps=1, ...) after rolling
        update.  Using preds[:, -1:] ensures both sides
        share shape (B, 1, ...) for a clean comparison.
        '''
        preds = simple_model_ones.advance(n=3)
        state_after = simple_model_ones.get_state()
        np.testing.assert_allclose(
            state_after["states_surface"],
            preds["states_surface"][:, -1:],
            rtol=0,
        )
        np.testing.assert_allclose(
            state_after["states_levels"],
            preds["states_levels"][:, -1:],
            rtol=0,
        )

    def test_step_tensors_require_grad_false(
            self,
    ) -> None:
        r'''All state tensors have requires_grad=False and no
        grad_fn after _step().

        Ensures that _step() runs under inference mode to
        prevent autograd graph retention and GPU memory
        bloat during autoregressive inference. Tests with a
        module that has learnable parameters to properly
        detect computation graph retention.
        '''
        # Arrange: Create a module with learnable parameters
        class ParameterizedModule(ForecastModule):
            def forward(
                    self,
                    states_surface: torch.Tensor,
                    states_levels: torch.Tensor,
                    **kwargs: object,
            ) -> tuple[torch.Tensor, torch.Tensor]:
                # Use a learnable parameter in the forward
                # pass to create a computation graph
                weight = torch.tensor(
                    2.0, dtype=states_surface.dtype,
                    requires_grad=True,
                )
                return (
                    states_surface[:, -1:] * weight,
                    states_levels[:, -1:] * weight,
                )

        param_module = ParameterizedModule(
            network=DummyNetwork(),
            pre_pipeline=PrePipeline(
                states_surface=IdentityPreModule(),
                states_levels=IdentityPreModule(),
            ),
            post_pipeline=PostPipeline(
                states_surface=IdentityPostModule(),
                states_levels=IdentityPostModule(),
            ),
        )
        model = ForecastModel(
            module=param_module,
            fabric=make_fabric(),
            dtype=torch.float32,
            compile=False,
        )
        model.set_state(_ones_state(batch=1))

        # Act
        model._autoregressive_step()

        # Assert
        for key, tensor in model.state.items():
            assert (
                tensor.requires_grad is False
            ), (
                f"State tensor '{key}' has "
                f"requires_grad=True after _step()"
            )
            assert (
                tensor.grad_fn is None
            ), (
                f"State tensor '{key}' has grad_fn="
                f"{tensor.grad_fn} after _step() "
                f"indicating retained computation graph"
            )


# -----------------------------------------------------------
# Unit tests
# -----------------------------------------------------------

class TestForecastModelUnittest:
    r'''Isolated unit tests for ForecastModel internals.'''

    def test_device_property_delegates_to_fabric(
            self, model: ForecastModel,
    ) -> None:
        r'''device property delegates to fabric.device.'''
        assert model.device == torch.device("cpu")

    def test_module_moved_to_dtype(
            self, simple_module: SimpleModule,
    ) -> None:
        r'''dtype=float64 moves parameters to float64.'''
        # Arrange + Act
        m = ForecastModel(
            module=simple_module,
            fabric=make_fabric(),
            dtype=torch.float64,
        )

        # Assert
        for p in m.module.parameters():
            assert p.dtype == torch.float64

    def test_get_state_returns_numpy(
            self, model: ForecastModel,
    ) -> None:
        r'''get_state returns numpy arrays.'''
        model.set_state(
            {"x": np.zeros((2,), dtype=np.float32)},
        )
        result = model.get_state()
        assert isinstance(result["x"], np.ndarray)

    def test_set_state_multiple_variables(
            self, model: ForecastModel,
    ) -> None:
        r'''Multiple variables round-trip through state.'''
        a = np.ones((2,), dtype=np.float32)
        b = np.full((3,), 5.0, dtype=np.float32)
        model.set_state({"a": a, "b": b})
        result = model.get_state()
        np.testing.assert_array_equal(result["a"], a)
        np.testing.assert_array_equal(result["b"], b)

    def test_set_auxiliary_stores_tensors(
            self, model: ForecastModel,
    ) -> None:
        r'''set_auxiliary converts numpy to torch tensors.'''
        data = np.zeros((2, 3), dtype=np.float32)
        model.set_auxiliary({"a": data})
        assert isinstance(
            model._auxiliary["a"], torch.Tensor,
        )

    def test_step_updates_state(
            self, model: ForecastModel,
    ) -> None:
        r'''After _step state values differ from initial.'''
        state = _surface_state()
        model.set_state(state)
        before = {
            k: v.copy()
            for k, v in model.get_state().items()
        }
        model._autoregressive_step()
        after = model.get_state()
        assert not np.allclose(
            before["states_surface"],
            after["states_surface"],
        )

    def test_step_with_auxiliary_passed_to_module(
            self, simple_module: SimpleModule,
    ) -> None:
        r'''Auxiliary dict forwarded as kwargs to forward.'''
        # Arrange
        mock_module = MagicMock(
            spec=SimpleModule,
            return_value=(
                torch.zeros(2, 1, 2, 4, 8),
                torch.zeros(2, 1, 2, 3, 4, 8),
            ),
        )
        mock_module.parameters = (
            simple_module.parameters
        )
        m = ForecastModel(
            module=simple_module,
            fabric=make_fabric(),
            dtype=torch.float32,
        )
        m._module = mock_module
        state = _surface_state()
        m.set_state(state)
        aux = {
            "mask": np.ones(
                (2, 1), dtype=np.float32,
            ),
        }
        m.set_auxiliary(aux)

        # Act
        m._autoregressive_step()

        # Assert
        call_kwargs = mock_module.call_args[1]
        assert "mask" in call_kwargs

    def test_step_with_forcings_passed_to_module(
            self, simple_module: SimpleModule,
    ) -> None:
        r'''Forcing dict forwarded as kwargs to forward.'''
        # Arrange
        mock_module = MagicMock(
            spec=SimpleModule,
            return_value=(
                torch.zeros(2, 1, 2, 4, 8),
                torch.zeros(2, 1, 2, 3, 4, 8),
            ),
        )
        mock_module.parameters = (
            simple_module.parameters
        )
        m = ForecastModel(
            module=simple_module,
            fabric=make_fabric(),
            dtype=torch.float32,
        )
        m._module = mock_module
        state = _surface_state()
        m.set_state(state)
        forcings = {
            "forcing_var": np.ones(
                (2, 4), dtype=np.float32,
            ),
        }

        # Act
        m._autoregressive_step(forcings=forcings)

        # Assert
        call_kwargs = mock_module.call_args[1]
        assert "forcing_var" in call_kwargs

    def test_step_preserves_state_keys(
            self, model: ForecastModel,
    ) -> None:
        r'''After _step the state dict retains same keys.'''
        state = _surface_state()
        model.set_state(state)
        keys_before = set(model.get_state().keys())
        model._autoregressive_step()
        keys_after = set(model.get_state().keys())
        assert keys_before == keys_after

    def test_advance_forcings_window_size_equals_n_in_plus_n_out(
            self, simple_module: SimpleModule,
    ) -> None:
        r'''Fixed window n_in+n_out passed at each step.

        With n_in_steps=1, n_out_steps=1, n_steps=3 and
        forcing shape (batch, 4, 4, 8) (= 1 + 3*1), each
        captured per-step window has shape[1] == 2
        (= n_in_steps + n_out_steps).
        '''
        n_steps = 3
        batch = 2
        m = _build_model_nsteps(n_in=1, n_out=1)
        state = _surface_state(batch=batch)
        m.set_state(state)
        forcing_arr = np.zeros(
            (batch, 4, 4, 8),
            dtype=np.float32,
        )
        forcings = {"fv": forcing_arr}

        captured = []
        original_step = ForecastModel._autoregressive_step

        def recording_step(
                self_m: ForecastModel,
                forcings: object = None,
        ) -> object:
            if forcings is not None:
                captured.append(
                    {
                        k: v.copy()
                        for k, v in forcings.items()
                    },
                )
            return original_step(
                self_m, forcings=forcings,
            )

        with patch.object(
            ForecastModel, "_autoregressive_step", recording_step,
        ):
            m.advance(n=n_steps, forcings=forcings)

        assert len(captured) == n_steps
        for rec in captured:
            assert rec["fv"].shape[1] == 2

    def test_single_step_window_step0(self) -> None:
        r'''n_in=n_out=1: step 0 gets window [0:2].

        Data (1, 6, 2) with n=5, n_in=n_out=1.
        Window formula: [i*n_out : i*n_out+n_in+n_out].
        Step 0: [0:2] -> tensor [[[0,1],[2,3]]].
        '''
        m = _build_recorder_model_n1()
        data = np.arange(
            12, dtype=np.float32,
        ).reshape(1, 6, 2)
        m.advance(
            n=5, forcings={"forcing_var": data},
        )
        expected = torch.tensor(
            [[[0.0, 1.0], [2.0, 3.0]]],
        )
        torch.testing.assert_close(
            m.module.recorded_forcing[0],
            expected,
        )

    def test_single_step_window_step4(self) -> None:
        r'''n_in=n_out=1: step 4 gets window [4:6].

        Data (1, 6, 2) with n=5, n_in=n_out=1.
        Step 4: [4:6] -> tensor [[[8,9],[10,11]]].
        '''
        m = _build_recorder_model_n1()
        data = np.arange(
            12, dtype=np.float32,
        ).reshape(1, 6, 2)
        m.advance(
            n=5, forcings={"forcing_var": data},
        )
        expected = torch.tensor(
            [[[8.0, 9.0], [10.0, 11.0]]],
        )
        torch.testing.assert_close(
            m.module.recorded_forcing[4],
            expected,
        )

    def test_multi_step_window_length(self) -> None:
        r'''n_in=n_out=1: every step gets window of 2.

        Data (1, 6, 2) with n=5, n_in=n_out=1.
        Fixed window size = n_in + n_out = 1 + 1 = 2.
        '''
        m = _build_recorder_model_n1()
        data = np.arange(
            12, dtype=np.float32,
        ).reshape(1, 6, 2)
        m.advance(
            n=5, forcings={"forcing_var": data},
        )
        for step, rec in enumerate(
            m.module.recorded_forcing,
        ):
            assert rec.shape[1] == 2, (
                f"Step {step}: expected window of 2"
                f", got {rec.shape[1]}"
            )

    def test_multi_step_window_advances_by_one(
            self,
    ) -> None:
        r'''Consecutive steps get windows shifted by 1.

        Data (1, 6, 2) with n=5, n_in=n_out=1:
          step 0 -> [:, 0:2] = [[[0,1],[2,3]]]
          step 4 -> [:, 4:6] = [[[8,9],[10,11]]]
        '''
        m = _build_recorder_model_n1()
        data = np.arange(
            12, dtype=np.float32,
        ).reshape(1, 6, 2)
        m.advance(
            n=5, forcings={"forcing_var": data},
        )
        recorded = m.module.recorded_forcing
        expected_0 = torch.tensor(
            [[[0.0, 1.0], [2.0, 3.0]]],
        )
        torch.testing.assert_close(
            recorded[0], expected_0,
        )
        expected_4 = torch.tensor(
            [[[8.0, 9.0], [10.0, 11.0]]],
        )
        torch.testing.assert_close(
            recorded[4], expected_4,
        )

    def test_n_in_steps_n_out_steps_stored(
            self,
    ) -> None:
        r'''n_in_steps and n_out_steps stored as attributes.

        ForecastModel with n_in_steps=3, n_out_steps=2 must
        expose those values as instance attributes after
        construction.
        '''
        m = _build_model_nsteps(n_in=3, n_out=2)

        assert m.n_in_steps == 3
        assert m.n_out_steps == 2


# -----------------------------------------------------------
# Error tests
# -----------------------------------------------------------

class TestForecastModelErrors:
    r'''Tests for expected error conditions.'''

    def test_state_raises_before_set(
            self, model: ForecastModel,
    ) -> None:
        r'''Accessing .state before set_state raises.'''
        with pytest.raises(ValueError):
            _ = model.state

    def test_forcing_shorter_than_n_raises_index_error(
            self,
    ) -> None:
        r'''Forcings with v.shape[1] < n raise IndexError.

        When v.shape[1]=3 and n=5 the window at step i=3
        is v[:, 3:2] which is empty; indexing [:, 0] inside
        the module raises IndexError.
        '''
        mod = _make_module(cls=ForcingConsumerModule)
        m = _build_model(mod)
        m.set_state(_ones_state(batch=1))
        short_forcing = {
            "forcing_var": np.ones(
                (1, 3, 2), dtype=np.float32,
            ),
        }
        with pytest.raises(IndexError):
            m.advance(n=5, forcings=short_forcing)

    def test_wrong_n_out_steps_raises_value_error(
            self, simple_module: SimpleModule,
    ) -> None:
        r'''Module returning wrong shape raises ValueError.

        With n_out_steps=1, a module returning (B, 2, ...)
        tensors must cause _step to raise ValueError with
        a message containing "n_out_steps=1".
        '''
        batch = 1
        surf_out = torch.zeros(batch, 2, 2, 4, 8)
        lev_out = torch.zeros(batch, 2, 2, 3, 4, 8)
        mock_mod = MagicMock(
            spec=SimpleModule,
            return_value=(surf_out, lev_out),
        )
        mock_mod.parameters = simple_module.parameters

        m = _build_model_nsteps(n_in=1, n_out=1)
        m._module = mock_mod
        m.set_state(_ones_state_n(n_time=1))

        with pytest.raises(
            ValueError, match="n_out_steps=1",
        ):
            m._autoregressive_step()

    def test_wrong_n_out_steps_message_contains_actual_size(
            self, simple_module: SimpleModule,
    ) -> None:
        r'''ValueError message includes actual output size.

        With n_out_steps=1 and a module returning (B, 2, ...),
        the error message must include "2" (the actual dim-1
        size) so the user can diagnose the mismatch.
        '''
        batch = 1
        surf_out = torch.zeros(batch, 2, 2, 4, 8)
        lev_out = torch.zeros(batch, 2, 2, 3, 4, 8)
        mock_mod = MagicMock(
            spec=SimpleModule,
            return_value=(surf_out, lev_out),
        )
        mock_mod.parameters = simple_module.parameters

        m = _build_model_nsteps(n_in=1, n_out=1)
        m._module = mock_mod
        m.set_state(_ones_state_n(n_time=1))

        with pytest.raises(ValueError, match="2"):
            m._autoregressive_step()

    def test_output_tuple_too_short_raises_value_error(
            self, simple_module: SimpleModule,
    ) -> None:
        r'''Too few tensors in output raises ValueError.

        The state has 2 keys (states_surface, states_levels).
        A module returning only 1 tensor must raise ValueError
        before _step touches output_tuple[1].
        '''
        surf_out = torch.zeros(1, 1, 2, 4, 8)
        mock_mod = MagicMock(
            spec=SimpleModule,
            return_value=(surf_out,),
        )
        mock_mod.parameters = simple_module.parameters

        m = _build_model_nsteps(n_in=1, n_out=1)
        m._module = mock_mod
        m.set_state(_ones_state_n(n_time=1))

        with pytest.raises(ValueError, match="2"):
            m._autoregressive_step()
            
    def test_output_non_tuple_raises_value_error(
            self, simple_module: SimpleModule,
    ) -> None:
        r'''Non-tuple output is normalized to tuple and validated.'''
        surf_out = torch.zeros(1, 1, 2, 4, 8)
        mock_mod = MagicMock(
            spec=SimpleModule,
            return_value=surf_out,
        )
        mock_mod.parameters = simple_module.parameters

        m = _build_model_nsteps(n_in=1, n_out=1)
        m._module = mock_mod
        m.set_state(_ones_state_n(n_time=1))

        with pytest.raises(ValueError, match="2"):
            m._autoregressive_step()

    def test_output_tuple_too_long_raises_value_error(
            self, simple_module: SimpleModule,
    ) -> None:
        r'''Too many tensors in output raises ValueError.

        The state has 2 keys. A module returning 3 tensors
        must raise ValueError; extras must not be silently
        ignored.
        '''
        surf_out = torch.zeros(1, 1, 2, 4, 8)
        lev_out = torch.zeros(1, 1, 2, 3, 4, 8)
        extra = torch.zeros(1, 1, 2, 4, 8)
        mock_mod = MagicMock(
            spec=SimpleModule,
            return_value=(surf_out, lev_out, extra),
        )
        mock_mod.parameters = simple_module.parameters

        m = _build_model_nsteps(n_in=1, n_out=1)
        m._module = mock_mod
        m.set_state(_ones_state_n(n_time=1))

        with pytest.raises(ValueError, match="2"):
            m._autoregressive_step()

    @pytest.mark.parametrize(
            (
                "n_in_steps",
                "n_out_steps",
                "match",
            ),
            [
                (1.5, 1, "n_in_steps"),
                (0, 1, "n_in_steps"),
                (1, 0, "n_out_steps"),
                (1, -1, "n_out_steps"),
            ],
    )
    def test_init_raises_for_invalid_n_steps(
            self,
            simple_module: SimpleModule,
            n_in_steps: object,
            n_out_steps: object,
            match: str,
    ) -> None:
        r'''Init with invalid n_in/n_out_steps raises ValueError.'''

        with pytest.raises(ValueError, match=match):
            ForecastModel(
                module=simple_module,
                fabric=make_fabric(),
                compile=False,
                n_in_steps=n_in_steps,
                n_out_steps=n_out_steps,
            )


# -----------------------------------------------------------
# Compile and Fabric tests
# -----------------------------------------------------------

class TestForecastModelCompileAndFabric:
    r'''Tests for torch.compile and Fabric integration.'''

    def test_compile_true_calls_torch_compile(
            self, simple_module: SimpleModule,
    ) -> None:
        r'''torch.compile is called before fabric.setup.'''
        # Arrange
        call_log: list = []
        mock_fabric = MagicMock(spec=lightning.fabric.Fabric)
        mock_fabric.device = torch.device("cpu")
        mock_fabric.to_device.side_effect = lambda t: t
        mock_fabric.setup.side_effect = (
            lambda m: call_log.append("fabric_setup") or m
        )

        with patch(
            "torch.compile",
            side_effect=lambda net, **kw: (
                call_log.append("compile") or net
            ),
        ):
            ForecastModel(
                module=simple_module,
                fabric=mock_fabric,
                compile=True,
            )

        # Assert — compile must precede fabric_setup
        assert (
            call_log.index("compile")
            < call_log.index("fabric_setup")
        )

    def test_fabric_setup_called_on_module(
            self, simple_module: SimpleModule,
    ) -> None:
        r'''ForecastModel calls fabric.setup once on module.'''
        # Arrange
        mock_fabric = MagicMock(spec=lightning.fabric.Fabric)
        mock_fabric.setup.return_value = simple_module
        mock_fabric.device = torch.device("cpu")
        mock_fabric.to_device.side_effect = lambda t: t

        # Act — module setter runs inside __init__
        ForecastModel(
            module=simple_module,
            fabric=mock_fabric,
        )

        # Assert
        assert mock_fabric.setup.call_count == 1
        assert (
            mock_fabric.setup.call_args.args[0]
            is simple_module
        )

    def test_step_uses_fabric_autocast(
            self, simple_module: SimpleModule,
    ) -> None:
        r'''_step uses fabric.autocast as context manager.'''
        # Arrange
        mock_fabric = MagicMock(spec=lightning.fabric.Fabric)
        mock_fabric.setup.return_value = simple_module
        mock_fabric.device = torch.device("cpu")
        mock_fabric.to_device.side_effect = lambda t: t

        model = ForecastModel(
            module=simple_module,
            fabric=mock_fabric,
        )
        model.set_state(_surface_state())

        # Act
        model._autoregressive_step()

        # Assert — context manager body is entered
        ctx = mock_fabric.autocast.return_value
        ctx.__enter__.assert_called_once()


# -----------------------------------------------------------
# Edge case tests
# -----------------------------------------------------------

class TestForecastModelEdgeCases:
    r'''Tests for edge cases and boundary conditions.'''

    def test_advance_state_updated(
            self, model: ForecastModel,
    ) -> None:
        r'''State after advance(2) differs from initial.'''
        state = _surface_state(batch=2)
        model.set_state(state)
        initial_surf = state["states_surface"].copy()
        model.advance(n=2)
        updated = model.get_state()
        assert not np.allclose(
            updated["states_surface"],
            initial_surf[:, -1],
        )

    def test_advance_output_shape_is_batch_n_then_spatial(
            self, simple_model_ones: ForecastModel,
    ) -> None:
        r'''advance(n=3) returns (batch, 3, vars, lat, lon).

        With torch.cat, SimpleModule output (B, 1, vars,
        lat, lon) per step is folded into the time axis,
        giving (B, n_calls * n_out_steps, vars, lat, lon).
        For n_out_steps=1 and n=3 this is (B, 3, vars, ...).
        '''
        preds = simple_model_ones.advance(n=3)
        assert preds["states_surface"].shape == (
            1, 3, 2, 4, 8,
        )
        assert preds["states_levels"].shape == (
            1, 3, 2, 3, 4, 8,
        )

    def test_advance_n1_returns_single_step(
            self, simple_model_ones: ForecastModel,
    ) -> None:
        r'''advance(n=1) produces one prediction of 2.0.

        Edge case: loop runs exactly once.  With torch.cat
        the trajectory folds the n_out_steps=1 time dim in,
        giving shape (B, 1, vars, lat, lon) not
        (B, 1, 1, vars, lat, lon).
        '''
        preds = simple_model_ones.advance(n=1)
        assert preds["states_surface"].shape == (
            1, 1, 2, 4, 8,
        )
        assert preds["states_levels"].shape == (
            1, 1, 2, 3, 4, 8,
        )
        np.testing.assert_allclose(
            preds["states_surface"][:, 0],
            2.0, rtol=0,
        )
        np.testing.assert_allclose(
            preds["states_levels"][:, 0],
            2.0, rtol=0,
        )

    def test_advance_n_out_2_trajectory_shape(
            self, simple_module: SimpleModule,
    ) -> None:
        r'''n_out_steps=2: advance(n=2) gives shape[1]==4.

        A mock module always returns (B, 2, ...) tensors.
        With n_out_steps=2 and n=2 calls the trajectory
        accumulates 2 * 2 = 4 time steps via torch.cat.
        '''
        batch = 1
        surf_out = torch.ones(batch, 2, 2, 4, 8)
        lev_out = torch.ones(batch, 2, 2, 3, 4, 8)
        mock_mod = MagicMock(
            spec=SimpleModule,
            return_value=(surf_out, lev_out),
        )
        mock_mod.parameters = simple_module.parameters

        m = _build_model_nsteps(n_in=1, n_out=2)
        m._module = mock_mod
        m.set_state(_ones_state_n(n_time=1))

        preds = m.advance(n=2)

        assert preds["states_surface"].shape[1] == 4
        assert preds["states_levels"].shape[1] == 4


# -----------------------------------------------------------
# Rolling window tests
# -----------------------------------------------------------

class TestForecastModelRollingWindow:
    r'''Tests for rolling-window state update semantics.'''

    def test_state_shape_preserved_after_advance(
            self,
    ) -> None:
        r'''Internal state shape[1] == n_in_steps after advance.

        With n_in_steps=2, n_out_steps=1, the rolling window
        must keep exactly 2 time steps in the internal state
        regardless of how many autoregressive steps run.
        '''
        m = _build_model_nsteps(n_in=2, n_out=1)
        m.set_state(_ones_state_n(n_time=2))

        m.advance(n=3)

        assert m.state["states_surface"].shape[1] == 2

    def test_rolling_window_drops_oldest_step(
            self, simple_module: SimpleModule,
    ) -> None:
        r'''After _step the oldest time step is evicted.

        n_in_steps=2, n_out_steps=1, mock module returning
        constant 99.0.  After one _step the new state[:, 0]
        equals old state[:, 1] (oldest dropped, newest
        appended).
        '''
        batch = 1
        surf_shape = (batch, 1, 2, 4, 8)
        lev_shape = (batch, 1, 2, 3, 4, 8)
        known_val = 99.0
        mock_mod = MagicMock(
            spec=SimpleModule,
            return_value=(
                torch.full(surf_shape, known_val),
                torch.full(lev_shape, known_val),
            ),
        )
        mock_mod.parameters = simple_module.parameters

        m = _build_model_nsteps(n_in=2, n_out=1)
        m._module = mock_mod

        init_state = {
            "states_surface": np.zeros(
                (batch, 2, 2, 4, 8), dtype=np.float32,
            ),
            "states_levels": np.zeros(
                (batch, 2, 2, 3, 4, 8),
                dtype=np.float32,
            ),
        }
        init_state["states_surface"][:, 0] = 0.0
        init_state["states_surface"][:, 1] = 1.0
        old_row1 = (
            init_state["states_surface"][:, 1].copy()
        )
        m.set_state(init_state)

        m._autoregressive_step()

        new_state = m.get_state()
        np.testing.assert_allclose(
            new_state["states_surface"][:, 0],
            old_row1,
            rtol=1e-5,
        )
        np.testing.assert_allclose(
            new_state["states_surface"][:, 1],
            known_val,
            rtol=1e-5,
        )

    def test_step_returns_prediction_tuple(
            self,
    ) -> None:
        r'''_step returns a tuple with shape[1]==n_out_steps.

        After the implementation change _step returns the
        module prediction tuple instead of None.  With
        n_out_steps=1 the returned surface tensor has
        shape[1] == 1.
        '''
        m = _build_model_nsteps(n_in=1, n_out=1)
        m.set_state(_ones_state_n(n_time=1))

        result = m._autoregressive_step()

        assert isinstance(result, tuple)
        assert result[0].shape[1] == 1

    def test_advance_accumulates_module_output_not_state(
            self,
    ) -> None:
        r'''Trajectory holds module outputs, not full state.

        With n_in_steps=2, n_out_steps=1, the internal state
        window has 2 entries but advance must accumulate only
        the 1-step prediction from each call.  SimpleModule
        doubles the last time step, so:
          call 0: output = 1.0 * 2 = 2.0
          call 1: output = 2.0 * 2 = 4.0
        Trajectory shape[1] == 2 with those values.
        '''
        m = _build_model_nsteps(n_in=2, n_out=1)
        m.set_state(_ones_state_n(n_time=2))

        preds = m.advance(n=2)

        assert preds["states_surface"].shape[1] == 2
        np.testing.assert_allclose(
            preds["states_surface"][:, 0], 2.0, rtol=0,
        )
        np.testing.assert_allclose(
            preds["states_surface"][:, 1], 4.0, rtol=0,
        )
