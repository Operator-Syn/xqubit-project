# New Notebook Cells — Paste After Cell 31 (Discussion of Initial Findings)

These cells extend the existing notebook with the new module features.
Each block maps to a new code/markdown cell in the Jupyter notebook.

---

## Cell A — Markdown (insert after id "31")

```markdown
## ⚙️ Extended Engine: PCA Threshold Sweep & Circuit Profiling

This section uses `quantum_engine_v2` to find the optimal PCA variance
threshold — too low loses signal; too high deepens the circuit and risks
**Barren Plateaus** on NISQ hardware.
```

---

## Cell B — Code (id "32b")

```python
import importlib
import src.quantum_engine_v2
importlib.reload(src.quantum_engine_v2)
from src.quantum_engine_v2 import PCAThresholdSweep, CircuitProfiler

# Run sweep over variance thresholds 0.80 → 0.99
sweep = PCAThresholdSweep(X_train_tfidf, n_qubits=2)   # X_train_tfidf from Cell 20
results = sweep.run()
best = PCAThresholdSweep.recommend(results)

print("--- PCA Threshold Sweep Results ---")
print(f"{'Threshold':>10}  {'n_components':>14}  {'Variance':>10}  {'Depth':>7}  {'OK?':>5}")
for r in results:
    tag = " ← RECOMMENDED" if r.recommended else ""
    print(f"{r.variance_threshold:>10.0%}  {r.n_components:>14}  "
          f"{r.explained_variance:>10.2%}  {r.profile.depth:>7}  "
          f"{'✓' if r.profile.depth <= 20 else '✗':>5}{tag}")
```

---

## Cell C — Code: Circuit Profile (id "32c")

```python
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes

fmap   = ZZFeatureMap(feature_dimension=2, reps=1)
ansatz = RealAmplitudes(num_qubits=2, reps=1)

profile = CircuitProfiler.profile(fmap, ansatz)
print("--- Circuit Profile ---")
print(profile.report())
```

---

## Cell D — Code: Quantum-Classical Delta (id "32d")

```python
from src.quantum_engine_v2 import compute_quantum_classical_delta

# y_pred_q   — from quantum VQC (Cell 27)
# y_pred_svm — from SVM baseline (Cell 29/30)
report = compute_quantum_classical_delta(
    y_test, y_pred_q, y_pred_svm,
    classes=["Greeting", "Farewell"]
)

print("--- Quantum-Classical Delta ---")
print(report.summary())
```

---

## Cell E — Code: Noisy Backend (IBM Brisbane sim)

```python
from src.quantum_engine_v2 import NoisySimBackend

noisy = NoisySimBackend(seed=42)
backend = noisy.get()
print("Noisy backend ready:", backend.name)
print("Noise channels by gate:", noisy.noise_summary)
```
