from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit_algorithms.optimizers import COBYLA
from qiskit_machine_learning.algorithms import VQC
# Updated imports for Qiskit 1.0+
from qiskit.primitives import StatevectorSampler as Sampler

def build_vqc(num_qubits=2, reps=2):
    """
    Constructs the Quantum Circuit components and the VQC object.
    Updated for Qiskit 1.x compatibility.
    """
    # 1. Feature Map: Encodes classical data [x, y] into quantum states
    feature_map = ZZFeatureMap(feature_dimension=num_qubits, reps=reps, entanglement='linear')

    # 2. Ansatz: The trainable 'Quantum Brain' (Hardware-efficient rotations)
    ansatz = RealAmplitudes(num_qubits=num_qubits, reps=reps)

    # 3. Optimizer: Classical algorithm to tune the circuit parameters
    optimizer = COBYLA(maxiter=100)

    # 4. Construct the Variational Quantum Classifier using the new Sampler
    sampler = Sampler()

    vqc = VQC(
        sampler=sampler,
        feature_map=feature_map,
        ansatz=ansatz,
        optimizer=optimizer
    )

    return vqc, feature_map, ansatz

def train_vqc(vqc, X_train, y_train):
    """
    Executes the training loop on the local statevector simulator.
    """
    print("🚀 Initializing Quantum Training Loop...")
    print(f"Training on {len(X_train)} linguistic samples...")

    # vqc.fit expects numpy arrays
    vqc.fit(X_train, y_train)

    print("✅ Quantum Model Training Complete.")
    return vqc
