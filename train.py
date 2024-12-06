import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import joblib
import os

# Load features and labels from disk
features = np.load('features.npy')
labels = np.load('labels.npy')

# Encode labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)
categorical_labels = to_categorical(encoded_labels)
joblib.dump(label_encoder, 'label_encoder.pkl')  # Save the label encoder

# Ensure all features have the same shape
max_length = max([f.shape[0] for f in features])
padded_features = np.array([np.pad(f, (0, max_length - f.shape[0]), mode='constant') for f in features])

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(padded_features, categorical_labels, test_size=0.2, random_state=42)

# Add channel dimension
X_train = X_train[..., np.newaxis]
X_test = X_test[..., np.newaxis]

# Define the model architecture
def create_model(input_shape):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Dropout(0.3),

        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.3),

        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.3),

        Flatten(),
        Dense(128, activation='relu', kernel_regularizer=l2(0.01)),  # Added L2 regularization
        Dropout(0.3),
        Dense(10, activation='softmax', kernel_regularizer=l2(0.01))  # Added L2 regularization
    ])
    return model

# Prompt user for training option
train_from_scratch = input("Type '1' to train from scratch, or '2' to load and resume training: ")

if train_from_scratch == '1':
    model = create_model((max_length, 1))
elif train_from_scratch == '2':
    if os.path.exists('voice_recognition_model.h5'):
        model = load_model('voice_recognition_model.h5')
    else:
        print("Trained model not found. Exiting.")
        exit()
else:
    print("Invalid choice. Exiting.")
    exit()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Callbacks
early_stopping_callback = EarlyStopping(
    monitor='val_loss',  # Monitor the validation loss
    patience=3,  # Number of epochs with no improvement before stopping
    restore_best_weights=True  # Restore the weights from the best epoch
)

checkpoint_callback = ModelCheckpoint(
    filepath='training_checkpoints/model-{epoch:04d}.h5',  # Adjusted filepath to end with .h5
    save_weights_only=False,
    monitor='val_loss',
    mode='min',
    save_best_only=True,
    verbose=1
)

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=15,  # Adjust the number of epochs as needed
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping_callback, checkpoint_callback]
)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy}")

# Save the model
model.save('voice_recognition_model.h5')