import os
import cv2
import numpy as np
from xml.etree import ElementTree
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

# Base path to your Pascal VOC dataset
BASE_PATH = r'C:\Users\Megha\OneDrive\Desktop\innovate intern\archive (2)'

# Paths to the train, val, and test sets
# Paths to the train, val, and test sets
TRAIN_PATH = os.path.join(BASE_PATH, 'Train-20200226T103300Z-001' , 'Train')
VAL_PATH = os.path.join(BASE_PATH, 'Val-20200226T103730Z-001' , 'Val')
TEST_PATH = os.path.join(BASE_PATH, 'Test-20200226T103653Z-001', 'Test')


TRAIN_FILE = os.path.join(TRAIN_PATH, 'train.txt')
VAL_FILE = os.path.join(VAL_PATH, 'val.txt')
TEST_FILE = os.path.join(TEST_PATH, 'test.txt')

# Function to parse annotation XML files
def parse_annotation(xml_file):
    try:
        tree = ElementTree.parse(xml_file)
    except FileNotFoundError:
        print(f"Annotation file not found: {xml_file}")
        return []
    
    root = tree.getroot()
    bboxes = []
    for obj in root.findall('object'):
        if obj.find('name').text == 'person':
            bbox = obj.find('bndbox')
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)
            bboxes.append((xmin, ymin, xmax, ymax))
    return bboxes

# Function to load images and their annotations with a limit
def load_data(data_path, file_path, limit=None):
    images = []
    labels = []
    count = 0
    with open(file_path, 'r') as f:
        for line in f:
            if limit and count >= limit:
                break
            img_id = line.strip()
            img_file = os.path.join(data_path, 'JPEGImages', f'{img_id}.jpg')
            xml_file = os.path.join(data_path, 'Annotations', f'{img_id}.xml')

            if not os.path.exists(img_file):
                print(f"Image file not found: {img_file}")
                continue

            if not os.path.exists(xml_file):
                print(f"Annotation file not found: {xml_file}")
                continue

            image = cv2.imread(img_file)
            if image is None:
                print(f"Failed to read image file: {img_file}")
                continue

            bboxes = parse_annotation(xml_file)
            for bbox in bboxes:
                xmin, ymin, xmax, ymax = bbox
                crop_img = image[ymin:ymax, xmin:xmax]
                resized_img = cv2.resize(crop_img, (64, 128))
                images.append(resized_img)
                labels.append(1)  # Pedestrian label
                # Add negative samples by random cropping
                h, w, _ = image.shape
                for _ in range(5):  # Add 5 negative samples per positive
                    x1 = np.random.randint(0, w-64)
                    y1 = np.random.randint(0, h-128)
                    neg_img = image[y1:y1+128, x1:x1+64]
                    images.append(neg_img)
                    labels.append(0)  # Non-pedestrian label
            count += 1
    return images, labels

# Function to compute HOG features with adjusted parameters
def compute_hog_features(image):
    winSize = (64, 128)
    blockSize = (16, 16)
    blockStride = (8, 8)
    cellSize = (8, 8)
    nbins = 9
    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
    return hog.compute(image).flatten()

# Function to prepare the dataset
def prepare_dataset(images, labels):
    features = [compute_hog_features(img) for img in images]
    features = np.array(features)
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    return features, np.array(labels)

# Load and preprocess the training data with a limit
train_images, train_labels = load_data(TRAIN_PATH, TRAIN_FILE, limit=100)  # Adjust limit as needed
train_features, train_labels = prepare_dataset(train_images, train_labels)

# Load and preprocess the validation data with a limit
val_images, val_labels = load_data(VAL_PATH, VAL_FILE, limit=50)  # Adjust limit as needed
val_features, val_labels = prepare_dataset(val_images, val_labels)

# Load and preprocess the testing data with a limit
test_images, test_labels = load_data(TEST_PATH, TEST_FILE, limit=50)  # Adjust limit as needed
test_features, test_labels = prepare_dataset(test_images, test_labels)

# Function to train an SVM model
def train_svm(features, labels):
    model = SVC(kernel='linear', C=1.0)
    model.fit(features, labels)
    return model

# Train the model
svm_model = train_svm(train_features, train_labels)

# Function to tune hyperparameters using grid search
def tune_hyperparameters(features, labels):
    param_grid = {'C': [0.1, 1, 10, 100]}
    grid_search = GridSearchCV(SVC(kernel='linear'), param_grid, cv=5)
    grid_search.fit(features, labels)
    return grid_search.best_params_

# Tune hyperparameters
best_params = tune_hyperparameters(train_features, train_labels)
print(f"Best Parameters: {best_params}")

# Train the model with the best parameters
svm_model = SVC(kernel='linear', C=best_params['C'])
svm_model.fit(train_features, train_labels)

# Function to evaluate the model
def evaluate_model(model, features, labels):
    predictions = model.predict(features)
    precision = precision_score(labels, predictions)
    recall = recall_score(labels, predictions)
    f1 = f1_score(labels, predictions)
    return precision, recall, f1

# Evaluate the model on validation and test sets
val_precision, val_recall, val_f1 = evaluate_model(svm_model, val_features, val_labels)
test_precision, test_recall, test_f1 = evaluate_model(svm_model, test_features, test_labels)

print(f'Validation - Precision: {val_precision}, Recall: {val_recall}, F1-score: {val_f1}')
print(f'Test - Precision: {test_precision}, Recall: {test_recall}, F1-score: {test_f1}')

# Function to plot metrics
def plot_metrics(metrics, title):
    labels = ['Precision', 'Recall', 'F1-score']
    plt.bar(labels, metrics)
    plt.ylim(0, 1)
    plt.title(title)
    plt.show()

# Plot validation and test metrics
plot_metrics([val_precision, val_recall, val_f1], 'Validation Metrics')
plot_metrics([test_precision, test_recall, test_f1], 'Test Metrics')



