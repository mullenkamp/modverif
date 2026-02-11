"""
Tests for model_eval.metrics module.
"""
import numpy as np
import pytest
from model_eval.metrics import (
    compute_ne,
    compute_ane,
    compute_rse,
    compute_bias,
    compute_pearson_correlation,
    compute_ne_domain,
    compute_ane_domain,
    compute_rmse_domain,
    compute_bias_domain,
    ContingencyTable,
)

class TestComputeNE:
    """Tests for compute_ne function."""
    def test_basic_calculation(self):
        source = np.array([100.0, 200.0, 50.0])
        test = np.array([110.0, 180.0, 75.0])
        result = compute_ne(source, test)
        expected = np.array([10, -10, 50], dtype=np.int16)
        np.testing.assert_array_equal(result, expected)

    def test_handles_zero_source(self):
        source = np.array([0.0, 1e-15, 100.0])
        test = np.array([10.0, 10.0, 110.0])
        result = compute_ne(source, test, epsilon=1e-10)
        assert result[0] == 0
        assert result[1] == 0
        assert result[2] == 10

class TestComputeANE:
    """Tests for compute_ane function."""
    def test_basic_calculation(self):
        source = np.array([100.0, 200.0, 50.0])
        test = np.array([110.0, 180.0, 75.0])
        result = compute_ane(source, test)
        expected = np.array([10, 10, 50], dtype=np.int16)
        np.testing.assert_array_equal(result, expected)

class TestComputeRSE:
    """Tests for compute_rse function."""
    def test_basic_calculation(self):
        source = np.array([100.0, 200.0, 50.0])
        test = np.array([110.0, 180.0, 75.0])
        result = compute_rse(source, test)
        expected = np.array([10.0, 20.0, 25.0], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected)

class TestComputeBias:
    """Tests for compute_bias function."""
    def test_basic_calculation(self):
        source = np.array([100.0, 200.0, 50.0])
        test = np.array([110.0, 180.0, 75.0])
        result = compute_bias(source, test)
        expected = np.array([10.0, -20.0, 25.0], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected)

class TestPearsonCorrelation:
    """Tests for compute_pearson_correlation function."""
    def test_perfect_correlation(self):
        x = np.array([1, 2, 3, 4, 5])
        y = x * 2 + 5
        assert compute_pearson_correlation(x, y) == pytest.approx(1.0)

    def test_negative_correlation(self):
        x = np.array([1, 2, 3, 4, 5])
        y = -x
        assert compute_pearson_correlation(x, y) == pytest.approx(-1.0)

class TestDomainMetrics:
    """Tests for domain-aggregated metrics."""
    def test_ne_domain(self):
        source = np.ones((1, 3, 3)) * 100
        test = np.ones((1, 3, 3)) * 110
        result = compute_ne_domain(source, test)
        assert result[0] == pytest.approx(10.0)

    def test_bias_domain(self):
        source = np.ones((1, 3, 3)) * 100
        test = np.ones((1, 3, 3)) * 110
        result = compute_bias_domain(source, test)
        assert result[0] == pytest.approx(10.0)

class TestContingencyTable:
    """Tests for ContingencyTable class and derived metrics."""
    @pytest.fixture
    def sample_table(self):
        # A=hits, B=false_alarms, C=misses, D=correct_negatives
        return ContingencyTable(hits=40, false_alarms=10, misses=20, correct_negatives=30)

    def test_from_data(self):
        source = np.array([0, 1, 2, 3])
        test = np.array([0, 2, 1, 3])
        threshold = 2
        # source_yes: [F, F, T, T]
        # test_yes:   [F, T, F, T]
        # Hits: source_yes & test_yes -> index 3 (1)
        # FA:   ~source_yes & test_yes -> index 1 (1)
        # Misses: source_yes & ~test_yes -> index 2 (1)
        # CN:   ~source_yes & ~test_yes -> index 0 (1)
        ct = ContingencyTable.from_data(source, test, threshold)
        assert ct.hits == 1
        assert ct.false_alarms == 1
        assert ct.misses == 1
        assert ct.correct_negatives == 1

    def test_pod(self, sample_table):
        # POD = 40 / (40 + 20) = 0.666...
        assert sample_table.pod() == pytest.approx(40/60)

    def test_far(self, sample_table):
        # FAR = 10 / (40 + 10) = 0.2
        assert sample_table.far() == pytest.approx(10/50)

    def test_csi(self, sample_table):
        # CSI = 40 / (40 + 10 + 20) = 40/70
        assert sample_table.csi() == pytest.approx(40/70)

    def test_gss(self, sample_table):
        # hits_random = (60 * 50) / 100 = 30
        # GSS = (40 - 30) / (40 + 10 + 20 - 30) = 10 / 40 = 0.25
        assert sample_table.gss() == pytest.approx(0.25)

    def test_bias(self, sample_table):
        # Bias = (40 + 10) / (40 + 20) = 50/60
        assert sample_table.bias() == pytest.approx(50/60)
