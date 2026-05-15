"""
tests/test_quantum_engine_v2.py
─────────────────────────────────────────────────────────────────────────────
Unit tests for src/quantum_engine_v2.py
Branch: feature/qnlu-modularization

Run with:
    python -m pytest tests/test_quantum_engine_v2.py -v
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from src.quantum_engine_v2 import (
    CircuitProfiler,
    CircuitProfile,
    PCAThresholdSweep,
    ThresholdResult,
    MultiClassVQC,
    VQCConfig,
    DeltaReport,
    compute_quantum_classical_delta,
)


# ═══════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture
def tiny_X():
    """32 samples of 50-dim TF-IDF-like features (random but seeded)."""
    rng = np.random.default_rng(0)
    X = np.abs(rng.standard_normal((32, 50)))
    # normalise rows like TF-IDF
    X = X / (X.sum(axis=1, keepdims=True) + 1e-9)
    return X


@pytest.fixture
def binary_labels():
    """16 greetings (0) + 16 farewells (1)."""
    return np.array([0] * 16 + [1] * 16)


@pytest.fixture
def multi_labels():
    """8 samples each of 4 classes."""
    return np.tile([0, 1, 2, 3], 8)


@pytest.fixture
def vqc_config():
    return VQCConfig(n_qubits=2, max_iter=5, seed=42, noisy=False)


# ═══════════════════════════════════════════════════════════════════════════
# CircuitProfiler
# ═══════════════════════════════════════════════════════════════════════════

class TestCircuitProfiler:

    def test_profile_returns_circuit_profile(self):
        from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
        fmap = ZZFeatureMap(feature_dimension=2, reps=1)
        ansatz = RealAmplitudes(num_qubits=2, reps=1)
        profile = CircuitProfiler.profile(fmap, ansatz)
        assert isinstance(profile, CircuitProfile)

    def test_profile_n_qubits_correct(self):
        from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
        fmap = ZZFeatureMap(feature_dimension=2, reps=1)
        ansatz = RealAmplitudes(num_qubits=2, reps=1)
        profile = CircuitProfiler.profile(fmap, ansatz)
        assert profile.n_qubits == 2

    def test_profile_depth_positive(self):
        from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
        fmap = ZZFeatureMap(feature_dimension=2, reps=1)
        ansatz = RealAmplitudes(num_qubits=2, reps=1)
        profile = CircuitProfiler.profile(fmap, ansatz)
        assert profile.depth > 0

    def test_profile_n_parameters_matches_ansatz(self):
        from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
        fmap = ZZFeatureMap(feature_dimension=2, reps=1)
        ansatz = RealAmplitudes(num_qubits=2, reps=1)
        expected_params = ansatz.num_parameters
        profile = CircuitProfiler.profile(fmap, ansatz)
        assert profile.n_parameters == expected_params

    def test_entanglement_ratio_in_range(self):
        from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
        fmap = ZZFeatureMap(feature_dimension=2, reps=1)
        ansatz = RealAmplitudes(num_qubits=2, reps=1)
        profile = CircuitProfiler.profile(fmap, ansatz)
        assert 0.0 <= profile.entanglement_ratio <= 1.0

    def test_report_is_string(self):
        from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
        fmap = ZZFeatureMap(feature_dimension=2, reps=1)
        ansatz = RealAmplitudes(num_qubits=2, reps=1)
        profile = CircuitProfiler.profile(fmap, ansatz)
        report = profile.report()
        assert isinstance(report, str)
        assert "Depth" in report


# ═══════════════════════════════════════════════════════════════════════════
# PCAThresholdSweep
# ═══════════════════════════════════════════════════════════════════════════

class TestPCAThresholdSweep:

    def test_run_returns_list(self, tiny_X):
        sweep = PCAThresholdSweep(tiny_X, n_qubits=2)
        results = sweep.run()
        assert isinstance(results, list)

    def test_run_length_matches_thresholds(self, tiny_X):
        sweep = PCAThresholdSweep(tiny_X, n_qubits=2)
        results = sweep.run()
        assert len(results) == len(PCAThresholdSweep.THRESHOLDS)

    def test_each_result_is_threshold_result(self, tiny_X):
        sweep = PCAThresholdSweep(tiny_X, n_qubits=2)
        for r in sweep.run():
            assert isinstance(r, ThresholdResult)

    def test_explained_variance_ascending(self, tiny_X):
        sweep = PCAThresholdSweep(tiny_X, n_qubits=2)
        results = sweep.run()
        variances = [r.explained_variance for r in results]
        assert variances == sorted(variances)

    def test_recommend_marks_exactly_one(self, tiny_X):
        sweep = PCAThresholdSweep(tiny_X, n_qubits=2)
        results = sweep.run()
        best = PCAThresholdSweep.recommend(results)
        recommended = [r for r in results if r.recommended]
        assert len(recommended) == 1
        assert recommended[0] is best

    def test_recommend_depth_within_nisq_limit(self, tiny_X):
        sweep = PCAThresholdSweep(tiny_X, n_qubits=2)
        results = sweep.run()
        best = PCAThresholdSweep.recommend(results)
        assert best.profile.depth <= PCAThresholdSweep.MAX_DEPTH_FOR_NISQ


# ═══════════════════════════════════════════════════════════════════════════
# MultiClassVQC
# ═══════════════════════════════════════════════════════════════════════════

class TestMultiClassVQC:

    def test_predict_before_fit_raises(self, vqc_config):
        model = MultiClassVQC(vqc_config, classes=["greeting", "farewell"])
        with pytest.raises(RuntimeError, match="fit"):
            model.predict(np.zeros((2, 2)))

    def test_circuit_summary_returns_profile(self, vqc_config):
        model = MultiClassVQC(vqc_config, classes=["greeting", "farewell"])
        profile = model.circuit_summary()
        assert isinstance(profile, CircuitProfile)
        assert profile.n_qubits == vqc_config.n_qubits

    def test_label_encoder_covers_all_classes(self, vqc_config):
        classes = ["greeting", "farewell", "thanks", "help"]
        model = MultiClassVQC(vqc_config, classes=classes)
        assert list(model.label_encoder.classes_) == classes

    def test_config_stored(self, vqc_config):
        model = MultiClassVQC(vqc_config, classes=["greeting", "farewell"])
        assert model.config is vqc_config


# ═══════════════════════════════════════════════════════════════════════════
# Quantum-Classical Delta
# ═══════════════════════════════════════════════════════════════════════════

class TestQuantumClassicalDelta:

    def _make_preds(self, size=40):
        rng = np.random.default_rng(7)
        y_true = rng.integers(0, 2, size=size)
        # Quantum slightly better than classical
        q = y_true.copy()
        flip_q = rng.choice(size, size=int(size * 0.25), replace=False)
        q[flip_q] ^= 1
        c = y_true.copy()
        flip_c = rng.choice(size, size=int(size * 0.35), replace=False)
        c[flip_c] ^= 1
        return y_true, q, c

    def test_returns_delta_report(self):
        y_true, y_q, y_c = self._make_preds()
        report = compute_quantum_classical_delta(y_true, y_q, y_c, ["greeting", "farewell"])
        assert isinstance(report, DeltaReport)

    def test_delta_is_q_minus_c(self):
        y_true, y_q, y_c = self._make_preds()
        report = compute_quantum_classical_delta(y_true, y_q, y_c, ["greeting", "farewell"])
        assert abs(report.delta - (report.quantum_accuracy - report.classical_accuracy)) < 1e-9

    def test_accuracies_in_range(self):
        y_true, y_q, y_c = self._make_preds()
        report = compute_quantum_classical_delta(y_true, y_q, y_c, ["greeting", "farewell"])
        assert 0.0 <= report.quantum_accuracy <= 1.0
        assert 0.0 <= report.classical_accuracy <= 1.0

    def test_precision_keys_match_classes(self):
        classes = ["greeting", "farewell"]
        y_true, y_q, y_c = self._make_preds()
        report = compute_quantum_classical_delta(y_true, y_q, y_c, classes)
        assert set(report.quantum_precision.keys()) == set(classes)
        assert set(report.classical_precision.keys()) == set(classes)

    def test_summary_contains_winner(self):
        y_true, y_q, y_c = self._make_preds()
        report = compute_quantum_classical_delta(y_true, y_q, y_c, ["greeting", "farewell"])
        summary = report.summary()
        assert "Winner" in summary
        assert ("Quantum" in summary or "Classical" in summary)

    def test_perfect_quantum_max_delta(self):
        y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        y_q    = y_true.copy()          # 100% accurate
        y_c    = 1 - y_true             # 0% accurate
        report = compute_quantum_classical_delta(y_true, y_q, y_c, ["greeting", "farewell"])
        assert report.quantum_accuracy == pytest.approx(1.0)
        assert report.classical_accuracy == pytest.approx(0.0)
        assert report.delta == pytest.approx(1.0)
