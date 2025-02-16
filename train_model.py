import os
import numpy as np
import pandas as pd
import h5py
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# ✅ Load dataset from local file (Ensure it's in the project folder)
dataset_path = os.path.join(os.getcwd(), "A_Z_Handwritten_Data.csv")

if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"❌ Dataset not found! Place it in: {dataset_path}")

# ✅ Read dataset
data = pd.read_csv(dataset_path, header=None).values

# ✅ Extract features and labels
X = data[:, 1:].reshape(-1, 28, 28, 1) / 255.0  # Normalize pixel values (0-1)
y = to_categorical(data[:, 0], num_classes=26)  # One-hot encoding (0-25 → A-Z)

# ✅ Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Build CNN Model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(26, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# ✅ Train the model
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), batch_size=64)

# ✅ Evaluate Model Performance
train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

print(f"✅ Training Accuracy: {train_acc * 100:.2f}%")
print(f"✅ Testing Accuracy: {test_acc * 100:.2f}%")

# ✅ Save the trained model
model_path = os.path.join(os.getcwd(), "alphabet_model.h5")
model.save(model_path)

print(f"✅ Model saved successfully at: {model_path}")

# ✅ Debugging: Test on one sample
index = np.random.randint(0, len(X_test))
sample_img = X_test[index].reshape(1, 28, 28, 1)
true_label = np.argmax(y_test[index])
prediction = model.predict(sample_img)
predicted_label = np.argmax(prediction)

plt.imshow(X_test[index].reshape(28, 28), cmap="gray")
plt.title(f"True: {chr(true_label + 65)}, Predicted: {chr(predicted_label + 65)}")
plt.show()
