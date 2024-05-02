from tqdm import tqdm
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

# Define constants
imageSize = 256
data_dir = "data"
num_classes = 4

def get_data(folder):
    """
    Load the data and labels from the given folder.
    """
    X = []
    y = []

    # Iterate through each class folder
    for folderName in tqdm(os.listdir(folder)):
        if not folderName.startswith('.'):
            # Assign labels based on folder names
            if folderName in ['NORMAL']:
                label = 0
            elif folderName in ['CNV']:
                label = 1
            elif folderName in ['DME']:
                label = 2
            elif folderName in ['DRUSEN']:
                label = 3
            else:
                continue

            class_path = os.path.join(folder, folderName)
            image_filenames = os.listdir(class_path)
            np.random.shuffle(image_filenames)  # Shuffle image filenames within each class

            # Iterate through each image in the class folder
            for image_filename in tqdm(image_filenames, leave=False):
                img_file = cv2.imread(os.path.join(class_path, image_filename), cv2.IMREAD_GRAYSCALE)
                if img_file is not None:
                    img_file = cv2.resize(img_file, (imageSize, imageSize))
                    img_file = img_file.reshape((imageSize, imageSize, 1))
                    X.append(img_file)
                    y.append(label)

    X = np.array(X)
    y = np.array(y)
    return X, y

# Load all data
X, y = get_data(data_dir)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# One-hot encode the labels
y_train_hot = to_categorical(y_train, num_classes=num_classes)
y_test_hot = to_categorical(y_test, num_classes=num_classes)

# Save the arrays to numpy files
np.save("X_train.npy", X_train)
np.save("y_train.npy", y_train)
np.save("X_test.npy", X_test)
np.save("y_test.npy", y_test)
np.save("y_train_hot.npy", y_train_hot)
np.save("y_test_hot.npy", y_test_hot)

# Print shapes of arrays
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train_hot shape:", y_train_hot.shape)
print("y_test_hot shape:", y_test_hot.shape)
