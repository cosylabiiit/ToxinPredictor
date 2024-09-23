import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report, roc_auc_score, precision_score, recall_score, roc_curve, matthews_corrcoef

print("Loading libraries and starting...")

# Load the dataset
df = pd.read_csv('merged_data_clean.csv')
df = df[df['Source'] == 'ToxiM']
print("Initial dataset loaded")
print(df.head(10))
print("Dataset shape:", df.shape)

# Store categorical columns and drop them
categorical_columns = ['SMILES', 'Source']
df = df.drop(columns=categorical_columns)
print("Dataset after dropping categorical columns")
print(df.head(10))

Y = df['Toxicity']

X_pca_df = pd.read_csv('X_pca_clean_with_boruta_ToxiM.csv')
print("PCA DataFrame shape:", X_pca_df.shape)
print(X_pca_df)

# Split data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X_pca_df, Y, test_size=0.2, random_state=42, shuffle=True, stratify=Y)
print("Data split into train and test sets")
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

# Check GPU availability
print("Num GPUs Available:", len(tf.config.experimental.list_physical_devices('GPU')))

# Define the model
model = Sequential()

model.add(Dense(512, input_dim=X_train.shape[1], kernel_regularizer=tf.keras.regularizers.l2(0.01)))
model.add(LeakyReLU(alpha=0.01))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(256, kernel_regularizer=tf.keras.regularizers.l2(0.01)))
model.add(LeakyReLU(alpha=0.01))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(128, kernel_regularizer=tf.keras.regularizers.l2(0.01)))
model.add(LeakyReLU(alpha=0.01))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Dense(64, kernel_regularizer=tf.keras.regularizers.l2(0.01)))
model.add(LeakyReLU(alpha=0.01))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Dense(32, kernel_regularizer=tf.keras.regularizers.l2(0.01)))
model.add(LeakyReLU(alpha=0.01))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Dense(16, kernel_regularizer=tf.keras.regularizers.l2(0.01)))
model.add(LeakyReLU(alpha=0.01))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
print("Model compiled")

# Define callbacks
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=0.0001)
early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

# Train the model with early stopping
print("Training the model...")
history = model.fit(X_train, Y_train, epochs=300, batch_size=32, validation_split=0.2, verbose=1, callbacks=[reduce_lr, early_stopping])

# Evaluate the model
print("Evaluating the model on test data...")
loss, accuracy = model.evaluate(X_test, Y_test, verbose=0)
print(f'Loss: {loss}, Accuracy: {accuracy}')

# Make predictions
print("Making predictions on test data...")
Y_pred_prob = model.predict(X_test)
Y_pred = (Y_pred_prob > 0.5).astype(int)

# Calculate additional metrics
f1 = f1_score(Y_test, Y_pred)
cm = confusion_matrix(Y_test, Y_pred)
cr = classification_report(Y_test, Y_pred)
roc_auc = roc_auc_score(Y_test, Y_pred_prob)
precision = precision_score(Y_test, Y_pred)
recall = recall_score(Y_test, Y_pred)
mcc = matthews_corrcoef(Y_test, Y_pred)

print("F1 Score: ", f1)
print("Confusion Matrix: \n", cm)
print("Classification Report: \n", cr)
print("ROC AUC Score: ", roc_auc)
print("Precision: ", precision)
print("Recall: ", recall)
print("Matthews Correlation Coefficient: ", mcc)
print("accuracy_score: ", accuracy_score(Y_test, Y_pred))

# Plot ROC curve
fpr, tpr, _ = roc_curve(Y_test, Y_pred_prob)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# Plotting the loss and accuracy curves
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy Curve')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()
