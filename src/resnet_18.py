import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, Conv2D, BatchNormalization, Activation, MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.preprocessing import label_binarize
from tensorflow.keras.applications.resnet import preprocess_input

# Set CUDA_VISIBLE_DEVICES to an empty string to disable GPU
os.environ['CUDA_VISIBLE_DEVICES'] = ''

save_dir = 'plots'
os.makedirs(save_dir, exist_ok=True)

# Load the data
X_train = np.load("X_train.npy", allow_pickle=True)
X_test = np.load("X_test.npy", allow_pickle=True)
y_train_hot = np.load("y_train_hot.npy", allow_pickle=True)
y_test_hot = np.load("y_test_hot.npy", allow_pickle=True)

# Print the dimensions of the loaded data
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train_hot shape:", y_train_hot.shape)
print("y_test_hot shape:", y_test_hot.shape)

# Check if ResNet-18 model is pretrained
def build_resnet18(input_shape, num_classes):
    # Input layer
    inputs = Input(shape=input_shape)

    # Convolutional block 1
    x = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    # Convolutional block 2
    x = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    # Convolutional block 3
    x = Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    # Convolutional block 4
    x = Conv2D(256, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    # Global average pooling and dense layer
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)

    # Output layer
    outputs = Dense(num_classes, activation='softmax')(x)

    # Create the model
    model = Model(inputs=inputs, outputs=outputs)

    return model

# Define the input shape and number of classes
input_shape = X_train.shape[1:]
num_classes = y_train_hot.shape[1]

# Build the ResNet-18 model
model = build_resnet18(input_shape, num_classes)

# Compile the model
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train_hot, epochs=8, batch_size=16, validation_data=(X_test, y_test_hot))

# Define the model name for saving the model and plots
model_name = 'ResNet18'

# Save the trained model
model.save(os.path.join(save_dir, f'{model_name}_model.h5'))

# Save the training history
np.save(os.path.join(save_dir, f'{model_name}_history.npy'), history.history)

# Make predictions
y_pred = model.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)
y_true_labels = np.argmax(y_test_hot, axis=1)

accuracy = accuracy_score(y_true_labels, y_pred_labels)
precision = precision_score(y_true_labels, y_pred_labels, average='weighted')
recall = recall_score(y_true_labels, y_pred_labels, average='weighted')
f1 = f1_score(y_true_labels, y_pred_labels, average='weighted')

print('\nEvaluation Metrics:')
print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')

# Plot confusion matrix
cm = confusion_matrix(y_true_labels, y_pred_labels)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, cmap='Reds', fmt='d', xticklabels=['NORMAL', 'CNV', 'DME', 'DRUSEN'],
            yticklabels=['NORMAL', 'CNV', 'DME', 'DRUSEN'])
plt.title(f'{model_name} Confusion Matrix', fontsize=10, fontweight='bold', pad=20)
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.grid(color='white', linestyle='-', linewidth=0.5)
plt.savefig(os.path.join(save_dir, f'{model_name}_confusion_matrix.png'), dpi=300)
plt.close()

# Plot training and validation accuracy
fig_acc = plt.figure(figsize=(6, 4))
plt.plot(range(1, len(history.history['accuracy']) + 1), history.history['accuracy'], color='red', linewidth=2, label='Train')
plt.plot(range(1, len(history.history['val_accuracy']) + 1), history.history['val_accuracy'], color='blue', linewidth=2, label='Validation')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.grid(color='gray', linestyle='-', linewidth=0.5)
plt.title(f'{model_name} Training and Validation Accuracy', fontweight='bold', fontsize=10, pad=20)
plt.savefig(os.path.join(save_dir, f'{model_name}_accuracy_plot.png'), dpi=300)
plt.close(fig_acc)

# Plot training and validation loss
fig_loss = plt.figure(figsize=(6, 4))
plt.plot(range(1, len(history.history['loss']) + 1), history.history['loss'], color='red', linewidth=2, label='Train')
plt.plot(range(1, len(history.history['val_loss']) + 1), history.history['val_loss'], color='blue', linewidth=2, label='Validation')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.grid(color='gray', linestyle='-', linewidth=0.5)
plt.title(f'{model_name} Training and Validation Loss', fontweight='bold', fontsize=10, pad=20)
plt.savefig(os.path.join(save_dir, f'{model_name}_loss_plot.png'), dpi=300)
plt.close(fig_loss)

# Plot ROC curve
fig_roc = plt.figure(figsize=(6, 4))
colors = ['blue', 'red', 'green', 'orange']
class_names = ['NORMAL', 'CNV', 'DME', 'DRUSEN']
y_test_hot = label_binarize(y_test_hot, classes=[0, 1, 2, 3])
for i in range(4):
    fpr, tpr, _ = roc_curve(y_test_hot[:, i], y_pred[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color=colors[i], linewidth=2, label=f'{class_names[i]} (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='black', linewidth=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.grid(color='gray', linestyle='-', linewidth=0.5)
plt.title(f'{model_name} ROC Curve', fontweight='bold', fontsize=10, pad=20)
plt.savefig(os.path.join(save_dir, f'{model_name}_roc_plot.png'), dpi=300)
plt.close(fig_roc)
