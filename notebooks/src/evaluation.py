import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

def evaluate_quantum_model(vqc, X_test, y_test, class_names=['Greeting', 'Farewell']):
    """
    Evaluates the VQC on unseen test data and visualizes the results.
    """
    print("🧪 Running evaluation on 40 unseen test samples...")

    # 1. Generate predictions from the Quantum Circuit
    y_pred = vqc.predict(X_test)

    # 2. Generate the textual report
    report = classification_report(y_test, y_pred, target_names=class_names)
    print("\n--- Quantum Classification Report ---")
    print(report)

    # 3. Create a Confusion Matrix visualization
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Intent')
    plt.ylabel('Actual Intent')
    plt.title('Quantum Intent Classification: Confusion Matrix')
    plt.show()

    return y_pred
