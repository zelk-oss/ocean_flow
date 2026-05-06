# -*- coding: utf-8 -*-
r'''Tests for src/forecast/config.py.

ForecastConfig and generate_forecast_configs.
'''

# External modules
import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

# Internal modules
from {{cookiecutter.project_slug}}.forecast.config import (
    ForecastConfig,
    generate_forecast_configs,
)
from {{cookiecutter.project_slug}}.forecast.runner import (
    _count_forecast_batches,
)

try:
    from {{cookiecutter.project_slug}}.forecast.config import (
        _total_init_ens_pairs,
    )
except ImportError:
    _total_init_ens_pairs = None


# -----------------------------------------------------------
# Helpers
# -----------------------------------------------------------


def _make_config(
    init_times,
    lead_times,
    ens_mems,
    n_store_freq,
):
    r'''Build a ForecastConfig directly from components.'''
    return ForecastConfig(
        init_times=pd.DatetimeIndex(init_times),
        lead_times=pd.TimedeltaIndex(lead_times),
        ens_mems=np.asarray(ens_mems),
        n_store_freq=n_store_freq,
    )


def _make_cfg(
    n_init=3,
    n_ens=2,
    batch_size=2,
    n_lead_steps=4,
    n_store_freq=2,
):
    r'''Return a minimal OmegaConf config for
    generate_forecast_configs.
    '''
    init_end = (
        pd.Timestamp("2020-01-01")
        + pd.Timedelta("6h") * (n_init - 1)
    ).isoformat()
    return OmegaConf.create({
        "init_start": "2020-01-01",
        "init_end": init_end,
        "init_freq": "6h",
        "lead_time": f"{6 * n_lead_steps}h",
        "step_freq": "6h",
        "ensemble_size": n_ens,
        "batch_size": batch_size,
        "n_store_freq": n_store_freq,
    })


# -----------------------------------------------------------
# Functional tests
# -----------------------------------------------------------


class TestForecastConfigFunctional:
    r'''End-to-end tests for generate_forecast_configs
    workflows.
    '''

    def test_yields_correct_number_of_configs(self):
        r'''3 init_times x 2 ens_mems / batch_size=2
        yields 3 configs.
        '''
        cfg = _make_cfg(
            n_init=3, n_ens=2, batch_size=2
        )
        configs = list(generate_forecast_configs(cfg))
        # cartesian product = 6 pairs; ceil(6/2) = 3
        assert len(configs) == 3

    def test_cartesian_product_order(self):
        r'''First config ens_mems reflects t0,t0 then
        t0,t1 (row-major).
        '''
        cfg = _make_cfg(
            n_init=2, n_ens=2, batch_size=2
        )
        configs = list(generate_forecast_configs(cfg))
        first = configs[0]
        # first batch: (t0,m0), (t0,m1)
        assert first.ens_mems[0] == 0
        assert first.ens_mems[1] == 1

    def test_lead_times_in_config(self):
        r'''lead_times in config matches
        pd.timedelta_range from cfg.
        '''
        cfg = _make_cfg(n_lead_steps=4)
        expected = pd.timedelta_range(
            start="6h", end="24h", freq="6h"
        )
        configs = list(generate_forecast_configs(cfg))
        assert configs[0].lead_times.equals(expected)

    def test_n_store_freq_propagated(self):
        r'''config.n_store_freq matches
        cfg.n_store_freq.
        '''
        cfg = _make_cfg(n_store_freq=3)
        configs = list(generate_forecast_configs(cfg))
        assert configs[0].n_store_freq == 3

    def test_pairs_are_init_times_major_single_batch(
        self,
    ):
        r'''Cartesian product is init-times-major within
        a single batch.

        For n_init=2, n_ens=3 the expected flat order is:
        (t0,0), (t0,1), (t0,2), (t1,0), (t1,1), (t1,2).
        '''
        cfg = _make_cfg(
            n_init=2, n_ens=3, batch_size=100
        )
        batches = list(
            generate_forecast_configs(cfg)
        )
        assert len(batches) == 1, (
            "Expected single batch for batch_size=100"
        )
        fc = batches[0]

        init_times = list(fc.init_times)
        ens_mems = list(fc.ens_mems)
        pairs = list(zip(init_times, ens_mems))

        t0 = pd.Timestamp("2020-01-01 00:00")
        t1 = pd.Timestamp("2020-01-01 06:00")
        expected = [
            (t0, 0), (t0, 1), (t0, 2),
            (t1, 0), (t1, 1), (t1, 2),
        ]
        assert pairs == expected

    def test_pairs_are_init_times_major_across_batches(
        self,
    ):
        r'''Cartesian product order is preserved when
        spread across batches.

        Uses batch_size=2 so pairs are split into three
        batches of 2. Concatenating the batches must
        still give init-times-major order.
        '''
        cfg = _make_cfg(
            n_init=2, n_ens=3, batch_size=2
        )
        all_init: list = []
        all_ens: list = []
        for fc in generate_forecast_configs(cfg):
            all_init.extend(list(fc.init_times))
            all_ens.extend(list(fc.ens_mems))

        t0 = pd.Timestamp("2020-01-01 00:00")
        t1 = pd.Timestamp("2020-01-01 06:00")
        expected_init = [
            t0, t0, t0, t1, t1, t1,
        ]
        expected_ens = [0, 1, 2, 0, 1, 2]
        assert all_init == expected_init
        assert all_ens == expected_ens

    def test_count_matches_generate_configs(
        self,
    ) -> None:
        r'''_count_forecast_batches matches the number of
        configs yielded by generate_forecast_configs.
        '''
        for n_init, n_ens, bs in [
            (3, 2, 2), (4, 3, 5),
            (1, 1, 1), (2, 4, 3),
        ]:
            cfg = _make_cfg(
                n_init=n_init, n_ens=n_ens,
                batch_size=bs,
            )
            count = _count_forecast_batches(cfg)
            configs = list(
                generate_forecast_configs(cfg)
            )
            assert count == len(configs), (
                f"Mismatch for n_init={n_init}, "
                f"n_ens={n_ens}, bs={bs}"
            )


# -----------------------------------------------------------
# Unit tests
# -----------------------------------------------------------


class TestForecastConfigUnittest:
    r'''Unit tests for ForecastConfig fields and
    iterators.
    '''

    @pytest.mark.skipif(
        _total_init_ens_pairs is None,
        reason="_total_init_ens_pairs not yet implemented",
    )
    def test_total_init_ens_pairs_basic(self) -> None:
        r'''_total_init_ens_pairs returns n_init * ens.'''
        # Arrange - 3 init_times, ensemble_size=2
        cfg_a = _make_cfg(
            n_init=3, n_ens=2, batch_size=1,
        )

        # Act
        result_a = _total_init_ens_pairs(cfg_a)

        # Assert
        assert result_a == 6

        # Arrange - 1 init_time, ensemble_size=1
        cfg_b = _make_cfg(
            n_init=1, n_ens=1, batch_size=1,
        )

        # Act
        result_b = _total_init_ens_pairs(cfg_b)

        # Assert
        assert result_b == 1

    def test_init_times_stored(self):
        r'''init_times is stored as the passed
        DatetimeIndex.
        '''
        idx = pd.DatetimeIndex(
            ["2020-01-01", "2020-01-02"]
        )
        cfg = _make_config(
            ["2020-01-01", "2020-01-02"],
            ["6h"], [0], 1,
        )
        assert cfg.init_times.equals(idx)

    def test_lead_times_stored(self):
        r'''lead_times is stored as the passed
        TimedeltaIndex.
        '''
        lt = pd.timedelta_range(
            start="6h", periods=3, freq="6h"
        )
        cfg = _make_config(
            ["2020-01-01"],
            ["6h", "12h", "18h"], [0], 1,
        )
        assert cfg.lead_times.equals(lt)

    def test_ens_mems_stored(self):
        r'''ens_mems is stored as the passed array.'''
        cfg = _make_config(
            ["2020-01-01"], ["6h"], [2, 5], 1
        )
        np.testing.assert_array_equal(
            cfg.ens_mems, [2, 5]
        )

    def test_n_store_freq_stored(self):
        r'''n_store_freq is stored as the passed
        integer.
        '''
        cfg = _make_config(
            ["2020-01-01"], ["6h"], [0], 7
        )
        assert cfg.n_store_freq == 7

    def test_yields_correct_number_of_chunks(self):
        r'''4 lead times with n_store_freq=2 yields
        exactly 2 chunks.
        '''
        lead_times = pd.timedelta_range(
            start="6h", periods=4, freq="6h"
        )
        cfg = _make_config(
            ["2020-01-01"], lead_times, [0],
            n_store_freq=2,
        )
        chunks = list(cfg.get_leadtime_iterator())
        assert len(chunks) == 2

    def test_chunk_sizes_equal_n_store_freq(self):
        r'''With n_store_freq=3 the first chunk contains
        3 entries.
        '''
        lead_times = pd.timedelta_range(
            start="6h", periods=6, freq="6h"
        )
        cfg = _make_config(
            ["2020-01-01"], lead_times, [0],
            n_store_freq=3,
        )
        chunks = list(cfg.get_leadtime_iterator())
        assert len(chunks[0]) == 3

    def test_single_lead_time(self):
        r'''A single lead time yields exactly one chunk
        of length 1.
        '''
        lead_times = pd.timedelta_range(
            start="6h", periods=1, freq="6h"
        )
        cfg = _make_config(
            ["2020-01-01"], lead_times, [0],
            n_store_freq=1,
        )
        chunks = list(cfg.get_leadtime_iterator())
        assert len(chunks) == 1
        assert len(chunks[0]) == 1


# -----------------------------------------------------------
# Error tests
# -----------------------------------------------------------


class TestForecastConfigErrors:
    r'''Error and contract tests for ForecastConfig.'''

    def test_direct_iteration_not_supported(self):
        r'''ForecastConfig is not directly iterable via
        Python protocol.
        '''
        lead_times = pd.timedelta_range(
            start="6h", periods=2, freq="6h"
        )
        cfg = _make_config(
            ["2020-01-01"], lead_times, [0],
            n_store_freq=1,
        )
        with pytest.raises(TypeError):
            list(cfg)


# -----------------------------------------------------------
# Edge case tests
# -----------------------------------------------------------


class TestForecastConfigEdgeCases:
    r'''Edge case tests for ForecastConfig and
    generate_forecast_configs.
    '''

    def test_last_chunk_smaller_when_not_divisible(
        self,
    ):
        r'''5 lead times with n_store_freq=3 yields a
        last chunk of length 2.
        '''
        lead_times = pd.timedelta_range(
            start="6h", periods=5, freq="6h"
        )
        cfg = _make_config(
            ["2020-01-01"], lead_times, [0],
            n_store_freq=3,
        )
        chunks = list(cfg.get_leadtime_iterator())
        assert len(chunks[-1]) == 2

    def test_batch_size_one(self):
        r'''n_init=2, n_ens=1, batch_size=1 gives
        2 configs each with 1 element.
        '''
        cfg = _make_cfg(
            n_init=2, n_ens=1, batch_size=1
        )
        configs = list(generate_forecast_configs(cfg))
        assert len(configs) == 2
        for c in configs:
            assert len(c.init_times) == 1

    def test_last_batch_contains_remainder_elements(
        self,
    ):
        r'''When batch_size does not divide
        n_init*n_ens, last batch has only the remaining
        elements and no elements are dropped.
        '''
        # 2 init * 3 ens = 6; batch_size=4 -> 4 + 2
        cfg = _make_cfg(
            n_init=2, n_ens=3, batch_size=4
        )
        batches = list(
            generate_forecast_configs(cfg)
        )

        assert len(batches) == 2
        assert len(batches[0].init_times) == 4
        assert len(batches[1].init_times) == 2

    def test_no_elements_dropped_when_batch_is_uneven(
        self,
    ):
        r'''Total number of (init_time, ens_mem) pairs
        equals n_init * n_ens regardless of whether
        batch_size evenly divides the total.
        '''
        cfg = _make_cfg(
            n_init=3, n_ens=4, batch_size=5
        )
        # 3 * 4 = 12; batch_size=5 -> 5, 5, 2
        all_pairs = []
        for fc in generate_forecast_configs(cfg):
            all_pairs.extend(
                zip(fc.init_times, fc.ens_mems)
            )

        assert len(all_pairs) == 12

    def test_last_batch_pairs_match_expected_tail(
        self,
    ):
        r'''The last (remainder) batch contains the
        correct trailing pairs.
        '''
        cfg = _make_cfg(
            n_init=2, n_ens=3, batch_size=4
        )
        batches = list(
            generate_forecast_configs(cfg)
        )
        last = batches[-1]

        t1 = pd.Timestamp("2020-01-01 06:00")
        assert list(last.init_times) == [t1, t1]
        assert list(last.ens_mems) == [1, 2]


# -----------------------------------------------------------
# DP world size tests (TDD: implementation pending)
# -----------------------------------------------------------


class TestDPWorldSize:
    r'''Tests for dp_world_size parameter in
    generate_forecast_configs.
    '''

    def test_dp_world_size_divides_batch_size(
        self,
    ) -> None:
        r'''batch_size=8, dp_world_size=4 produces configs
        with 2 pairs each.
        '''
        # Arrange
        cfg = _make_cfg(
            n_init=4, n_ens=2, batch_size=8,
        )

        # Act
        configs = list(
            generate_forecast_configs(
                cfg, dp_world_size=4,
            )
        )

        # Assert -- local batch = 8 // 4 = 2
        for fc in configs:
            assert len(fc.init_times) <= 2
            assert len(fc.ens_mems) <= 2

    def test_dp_world_size_default_preserves_behavior(
        self,
    ) -> None:
        r'''dp_world_size=1 produces the same configs as
        calling without the parameter.
        '''
        # Arrange
        cfg = _make_cfg(
            n_init=3, n_ens=2, batch_size=2,
        )

        # Act
        configs_default = list(
            generate_forecast_configs(cfg)
        )
        configs_explicit = list(
            generate_forecast_configs(
                cfg, dp_world_size=1,
            )
        )

        # Assert
        assert len(configs_default) == len(
            configs_explicit
        )
        for a, b in zip(
            configs_default, configs_explicit
        ):
            assert list(a.init_times) == list(
                b.init_times
            )
            assert list(a.ens_mems) == list(
                b.ens_mems
            )

    def test_dp_world_size_floor_division(
        self,
    ) -> None:
        r'''batch_size=7, dp_world_size=4 produces configs
        with max(1, 7//4)=1 pair each.
        '''
        # Arrange
        cfg = _make_cfg(
            n_init=3, n_ens=1, batch_size=7,
        )

        # Act
        configs = list(
            generate_forecast_configs(
                cfg, dp_world_size=4,
            )
        )

        # Assert -- local batch = max(1, 7//4) = 1
        for fc in configs:
            assert len(fc.init_times) <= 1
            assert len(fc.ens_mems) <= 1
        # Total pairs preserved
        total = sum(
            len(fc.init_times) for fc in configs
        )
        assert total == 3
