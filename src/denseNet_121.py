import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.preprocessing import label_binarize

# Set CUDA_VISIBLE_DEVICES to an empty string to disable GPU
os.environ['CUDA_VISIBLE_DEVICES'] = ''

save_dir = 'plots'
os.makedirs(save_dir, exist_ok=True)

# Load the data
X_train = np.load("X_train.npy", allow_pickle=True)
X_test = np.load("X_test.npy", allow_pickle=True)
y_train_hot = np.load("y_train_hot.npy", allow_pickle=True)
y_test_hot = np.load("y_test_hot.npy", allow_pickle=True)

# Convert grayscale images to RGB
X_train_rgb = np.concatenate([X_train, X_train, X_train], axis=-1)
X_test_rgb = np.concatenate([X_test, X_test, X_test], axis=-1)

# Preprocess the images
X_train_rgb = X_train_rgb / 255.0
X_test_rgb = X_test_rgb / 255.0

# Load the pre-trained DenseNet121 model without the top layer
base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(256, 256, 3))

# Freeze the layers in the base model
for layer in base_model.layers:
    layer.trainable = False

# Add a global average pooling layer and a fully connected layer with dropout regularization
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)  # Dropout regularization with a rate of 0.5
predictions = Dense(4, activation='softmax')(x)

# Create the final model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train_rgb, y_train_hot, epochs=8, batch_size=16, validation_data=(X_test_rgb, y_test_hot))

# Plot training and validation accuracy using Seaborn
sns.set(style='whitegrid')
plt.figure(figsize=(8, 6))
sns.lineplot(range(1, len(history.history['accuracy']) + 1), history.history['accuracy'], label='Train')
sns.lineplot(range(1, len(history.history['val_accuracy']) + 1), history.history['val_accuracy'], label='Validation')
plt.title('DenseNet121 Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.savefig(os.path.join(save_dir, 'DenseNet121_accuracy_plot.png'))
plt.close()

# Plot training and validation loss using Seaborn
plt.figure(figsize=(8, 6))
sns.lineplot(range(1, len(history.history['loss']) + 1), history.history['loss'], label='Train')
sns.lineplot(range(1, len(history.history['val_loss']) + 1), history.history['val_loss'], label='Validation')
plt.title('DenseNet121 Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.savefig(os.path.join(save_dir, 'DenseNet121_loss_plot.png'))
plt.close()

# Use the model to predict the labels for the validation set
y_pred = model.predict(X_test_rgb)

# Convert the predicted labels and true labels to one-dimensional arrays
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test_hot, axis=1)

# Calculate evaluation metrics
accuracy = accuracy_score(y_true_classes, y_pred_classes)
precision = precision_score(y_true_classes, y_pred_classes, average='weighted')
recall = recall_score(y_true_classes, y_pred_classes, average='weighted')
f1 = f1_score(y_true_classes, y_pred_classes, average='weighted')

print('\nEvaluation Metrics:')
print(f'Accuracy of DenseNet121: {accuracy:.4f}')
print(f'Precision of DenseNet121: {precision:.4f}')
print(f'Recall of DenseNet121: {recall:.4f}')
print(f'F1 Score of DenseNet121: {f1:.4f}')

# Set the DPI value for higher quality plots
dpi = 300

# Plot training and validation accuracy using Seaborn
plt.figure(figsize=(8, 6), dpi=dpi)
sns.lineplot(range(1, len(history.history['accuracy']) + 1), history.history['accuracy'], label='Train', color='red', linewidth=2.5)
sns.lineplot(range(1, len(history.history['val_accuracy']) + 1), history.history['val_accuracy'], label='Validation', color='blue', linewidth=2.5)
plt.title('DenseNet121 Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.savefig(os.path.join(save_dir, 'DenseNet121_accuracy_plot.png'), dpi=dpi)
plt.close()

# Plot training and validation loss using Seaborn
plt.figure(figsize=(8, 6), dpi=dpi)
sns.lineplot(range(1, len(history.history['loss']) + 1), history.history['loss'], label='Train', color='red', linewidth=2.5)
sns.lineplot(range(1, len(history.history['val_loss']) + 1), history.history['val_loss'], label='Validation', color='blue', linewidth=2.5)
plt.title('DenseNet121 Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.savefig(os.path.join(save_dir, 'DenseNet121_loss_plot.png'), dpi=dpi)
plt.close()

# Generate and save the confusion matrix with purple color gradient
plt.figure(figsize=(8, 6), dpi=dpi)
sns.heatmap(cm, annot=True, cmap='Purples', fmt='d', xticklabels=['NORMAL', 'CNV', 'DME', 'DRUSEN'],
            yticklabels=['NORMAL', 'CNV', 'DME', 'DRUSEN'])
plt.title('DenseNet121 Confusion Matrix')
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.savefig(os.path.join(save_dir, 'DenseNet121_confusion_matrix.png'), dpi=dpi)
plt.close()

# Generate and save the ROC curve plot
plt.figure(figsize=(8, 6), dpi=dpi)
colors = ['blue', 'red', 'green', 'orange']
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label='ROC curve of class {0} (AUC = {1:.2f})'.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
plt.xlim([-0.05, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('DenseNet121 Receiver Operating Characteristic (ROC)')
plt.legend(loc='lower right')
plt.savefig(os.path.join(save_dir, 'DenseNet121_roc_curve.png'), dpi=dpi)
plt.close()