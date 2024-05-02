import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
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

# Load the pre-trained InceptionResNetV2 model without the top layer
base_model = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(256, 256, 3))

# Freeze the layers in the base model
for layer in base_model.layers:
    layer.trainable = False

# Add a global average pooling layer and a fully connected layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
predictions = Dense(4, activation='softmax')(x)

# Create the final model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train_rgb, y_train_hot, epochs=8, batch_size=16, validation_data=(X_test_rgb, y_test_hot))

# Define the model name for saving the model and plots
model_name = 'InceptionResNetV2'

# Save the trained model
model.save(os.path.join(save_dir, f'{model_name}_model.h5'))

# Save the training history
np.save(os.path.join(save_dir, f'{model_name}_history.npy'), history.history)

y_pred = model.predict(X_test_rgb)

# Convert the predicted labels and true labels to one-dimensional arrays
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test_hot, axis=1)

accuracy = accuracy_score(y_true_classes, y_pred_classes)
precision = precision_score(y_true_classes, y_pred_classes, average='weighted')
recall = recall_score(y_true_classes, y_pred_classes, average='weighted')
f1 = f1_score(y_true_classes, y_pred_classes, average='weighted')

print('\nEvaluation Metrics:')
print(f'Accuracy of {model_name}: {accuracy:.4f}')
print(f'Precision of {model_name}: {precision:.4f}')
print(f'Recall of {model_name}: {recall:.4f}')
print(f'F1 Score of {model_name}: {f1:.4f}')

# Set the DPI value for higher quality plots
dpi = 300

# Function to format and save the plots
def save_plot(fig, title, save_dir, file_name):
    fig.set_size_inches(8, 6)  # Set square dimensions
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.95)  # Set title properties
    fig.savefig(os.path.join(save_dir, file_name), dpi=dpi)
    plt.close(fig)

cm = confusion_matrix(y_true_classes, y_pred_classes)
plt.figure(figsize=(8, 6), dpi=dpi, facecolor='white')  # Set facecolor to white
sns.heatmap(cm, annot=True, cmap='Reds', fmt='d', xticklabels=['NORMAL', 'CNV', 'DME', 'DRUSEN'],
            yticklabels=['NORMAL', 'CNV', 'DME', 'DRUSEN'])
plt.title(f'{model_name} Confusion Matrix', fontweight='bold', loc='center')
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.savefig(os.path.join(save_dir, f'{model_name}_confusion_matrix.png'), dpi=dpi, facecolor='white')  # Set facecolor to white for saving
plt.close()

# Plot training and validation accuracy
fig_acc = plt.figure()
plt.plot(range(1, len(history.history['accuracy']) + 1), history.history['accuracy'], color='red', linewidth=2, label='Train')
plt.plot(range(1, len(history.history['val_accuracy']) + 1), history.history['val_accuracy'], color='blue', linewidth=2, label='Validation')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
save_plot(fig_acc, f'{model_name} Model Accuracy', save_dir, f'{model_name}_accuracy_plot.png')

# Plot training and validation loss
fig_loss = plt.figure()
plt.plot(range(1, len(history.history['loss']) + 1), history.history['loss'], color='red', linewidth=2, label='Train')
plt.plot(range(1, len(history.history['val_loss']) + 1), history.history['val_loss'], color='blue', linewidth=2, label='Validation')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
save_plot(fig_loss, f'{model_name} Model Loss', save_dir, f'{model_name}_loss_plot.png')

# Calculate the False Positive Rate (FPR), True Positive Rate (TPR), and Area Under the Curve (AUC) for each class
y_pred = model.predict(X_test_rgb)
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(4):
    fpr[i], tpr[i], _ = roc_curve(y_test_hot[:, i], y_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC curves
fig_roc = plt.figure()
colors = ['blue', 'red', 'green', 'orange']
class_names = ['NORMAL', 'CNV', 'DME', 'DRUSEN']
for i in range(4):
    plt.plot(fpr[i], tpr[i], color=colors[i], linewidth=2, label=f'{class_names[i]} (AUC = {roc_auc[i]:.2f})')
plt.plot([0, 1], [0, 1], color='black', linewidth=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
save_plot(fig_roc, f'{model_name} ROC Curve', save_dir, f'{model_name}_roc_plot.png')
