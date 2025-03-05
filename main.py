import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("dataset/heart_disease_risk_dataset_earlymed.csv")

# Split features and target
X = df.drop(columns=["Heart_Risk"])
y = df["Heart_Risk"]

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape for CNN input
X_train_cnn = X_train[..., np.newaxis]
X_test_cnn = X_test[..., np.newaxis]

# Model 1: 1D CNN + ImageNet
cnn_imagenet = tf.keras.Sequential([
    tf.keras.layers.Conv1D(32, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
cnn_imagenet.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Model 2: 1D CNN + ResNet
cnn_resnet = tf.keras.Sequential([
    tf.keras.layers.Conv1D(64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)),
    tf.keras.layers.Conv1D(64, kernel_size=3, activation='relu'),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
cnn_resnet.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Model 3: Transformer (MLP-Mixer)
mlp_mixer = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
mlp_mixer.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training function
def train_model(model, X_train, y_train, X_test, y_test, name):
    print(f"Training {name}...")
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=32)
    return history

# Train models
history_cnn_imagenet = train_model(cnn_imagenet, X_train_cnn, y_train, X_test_cnn, y_test, "CNN + ImageNet")
history_cnn_resnet = train_model(cnn_resnet, X_train_cnn, y_train, X_test_cnn, y_test, "CNN + ResNet")
history_mlp_mixer = train_model(mlp_mixer, X_train, y_train, X_test, y_test, "Transformer (MLP-Mixer)")

# Evaluate models
def evaluate_model(model, X_test, y_test, name):
    y_pred = (model.predict(X_test) > 0.5).astype(int)
    accuracy = np.mean(y_pred.ravel() == y_test.to_numpy().ravel())
    print(f"\n{name} Performance:")
    print(classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {name}')
    plt.show()
    return accuracy

# Store accuracies
accuracies = {
    "CNN + ImageNet": evaluate_model(cnn_imagenet, X_test_cnn, y_test, "CNN + ImageNet"),
    "CNN + ResNet": evaluate_model(cnn_resnet, X_test_cnn, y_test, "CNN + ResNet"),
    "Transformer (MLP-Mixer)": evaluate_model(mlp_mixer, X_test, y_test, "Transformer (MLP-Mixer)")
}

# Select best model
best_model_name = max(accuracies, key=accuracies.get)
print(f"Best model: {best_model_name} with accuracy {accuracies[best_model_name]:.4f}")

# Save best model
best_model = {
    "CNN + ImageNet": cnn_imagenet,
    "CNN + ResNet": cnn_resnet,
    "Transformer (MLP-Mixer)": mlp_mixer
}[best_model_name]

best_model.save(f"best_model_{best_model_name.replace(' ', '_')}.h5")

# Visualization function
def plot_training_history(histories, names):
    plt.figure(figsize=(10, 5))
    for history, name in zip(histories, names):
        plt.plot(history.history['accuracy'], label=f'{name} Train Acc')
        plt.plot(history.history['val_accuracy'], label=f'{name} Val Acc', linestyle='dashed')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training Accuracy Comparison')
    plt.show()

# Plot training history
plot_training_history([history_cnn_imagenet, history_cnn_resnet, history_mlp_mixer],
                      ["CNN + ImageNet", "CNN + ResNet", "Transformer (MLP-Mixer)"])
