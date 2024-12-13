# OpenCV_BASICS_MEXE4101_Abante_Potestades
# Extracting Contours for Analyzing Shape Features in Children's Hand-drawn Shapes

## Introduction

Analyzing children's hand-drawn shapes has significant applications in computer vision, including educational assessment, developmental psychology, and art analysis. By leveraging contours to detect and analyze shape features, we can gain insights into the underlying patterns and characteristics of these drawings. This approach can aid in understanding motor skills, cognitive development, and creativity in children. Additionally, the method can support educators and researchers in evaluating drawing accuracy and complexity. The use of contour analysis provides a non-invasive and efficient tool for assessing developmental milestones through visual inputs.

## Abstract

This project aims to analyze children's hand-drawn shapes using contour detection techniques. Contours are used to extract shape features, enabling the identification and categorization of geometric and irregular shapes. The project employs OpenCV for preprocessing, contour extraction, and feature analysis. Expected outcomes include a robust method for classifying shapes, insights into drawing tendencies, and a platform for further developmental studies. By focusing on contour-based techniques, the project provides a cost-effective and scalable solution for analyzing visual data. The approach emphasizes accuracy and adaptability, making it suitable for diverse applications in education and research.

## Methodology

This is the step-by-step methodology for Extracting Contours for Analyzing Shape Features in Children's Hand-drawn Shapes in google colab:

### Project Methods
#### 1. Set Up the Environment:

- Install necessary libraries: OpenCV, NumPy, and other required Python packages.
- Configure the development environment (e.g., Jupyter Notebook, Google Colab, or a local Python environment).
- Verify the installation of dependencies and ensure compatibility with the project requirements.
  
#### 2. Data Collection:

- Utilize the Hand-drawn Shapes (HDS) Dataset from Kaggle, which includes a diverse mix of geometric and irregular shapes created by children.
- Digitize additional drawings if needed using a scanner or camera.

#### 3. Preprocessing:

- Convert images to grayscale using OpenCV’s cv2.cvtColor function to simplify processing.
- Use thresholding (e.g., cv2.threshold) to binarize the images.

#### 4. Contour Detection:

- Detect contours in binarized images using cv2.findContours.
- Filter and sort contours based on area, perimeter, or hierarchy to focus on relevant shapes.

#### 5. Feature Extraction:

- Calculate shape features such as aspect ratio, extent, solidity, and eccentricity.

#### 6. Classification:

- Group shapes into categories (e.g., geometric, irregular) using clustering algorithms or predefined thresholds.
- Optionally, train a machine learning model with labeled data for automated classification.


#### 7. Visualization:

- Overlay detected contours on the original images using cv2.drawContours.
- Display shape metrics (e.g., area, perimeter) as annotations.

#### 8. Evaluation:

- Validate the method using metrics such as accuracy, precision, and recall for shape classification.
- Test with diverse datasets to ensure robustness and generalization.


## Conclusion

This project successfully implemented a method to extract and analyze shape features in children’s hand-drawn shapes using contour detection. Key findings include:

#### Findings:

- Contour-based methods effectively identify and classify geometric and irregular shapes.
- Shape features such as aspect ratios provide meaningful metrics for categorization.

#### Challenges:

- Handling overlapping or poorly defined shapes required additional preprocessing.
- Noise and variations in drawing styles posed difficulties in maintaining high classification accuracy.

#### Outcomes:

- Developed a pipeline for contour-based analysis of hand-drawn shapes.
- Highlighted areas for improvement, such as integrating deep learning for enhanced feature extraction and classification.

This project demonstrates the potential of computer vision techniques in analyzing creative outputs, providing a foundation for future studies in education and developmental psychology.

### Additional Materials

### CODE

```python
from google.colab import drive
drive.mount('/content/drive')

```
```python
from google.colab import files
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from PIL import Image
import os
```
```python
# 1. Feature Extraction Function

def image_to_feature_vector(image_path):
    image = Image.open(image_path)
    # Resize the image to match training data dimensions
    image = image.resize((70, 70))  # Assuming training images were 70x70
    image_array = np.array(image)
    feature_vector = image_array.flatten()
    return feature_vector
```
```python
# 2. Label Mapping
label_mapping = {'triangle': 0, 'rectangle': 1, 'ellipse': 2, 'other': 3}

def map_labels_to_int(labels, label_mapping):
    return np.array([label_mapping[label] for label in labels])
```
```python
# 3. Data Loading Function
def load_image_data(folder_path):
    image_data = []
    labels = []
    for dirname, foldername, filenames in os.walk(folder_path):
        for filename in filenames:
            # Check if the file is a PNG image and the name (without extension) is in label_mapping
            if filename.endswith(".png") and filename.split(".")[0] in label_mapping:
                image_path = os.path.join(dirname, filename)

                # Attempt to load the image
                try:
                    feature_vector = image_to_feature_vector(image_path)
                    label = filename.split(".")[0]
                    image_data.append(feature_vector)
                    labels.append(label_mapping[label])
                except Exception as e:
                    # Print an error message if image loading fails
                    print(f"Error loading image {image_path}: {e}")

    # Check if any images were loaded, print a warning if not
    if not image_data:
        print("Warning: No images were loaded. Check the folder path and image names.")

    return np.array(image_data), np.array(labels)
```
```python
# 4. Shape Prediction Function (Modified)
def predict_shape(image_path, knn_model):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    shape_name = ""
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        perimeter = cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, 0.04 * perimeter, True)

        if len(approx) == 3:
            shape_name = "triangle"
        elif len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w) / h
            if 0.8 <= aspect_ratio <= 1.2:
                shape_name = "square"  # Consider as square
            else:
                shape_name = "rectangle"
        elif len(approx) >= 6:  # Potential circle or ellipse
            # Calculate circularity to distinguish between circle and ellipse
            area = cv2.contourArea(largest_contour)
            circularity = 4 * np.pi * (area / (perimeter ** 2))

            if 0.8 <= circularity <= 1.2:  # Adjust threshold as needed
                shape_name = "circle"
            else:
                shape_name = "ellipse"

        return shape_name

    else:
        # KNN fallback (Modified)
        feature_vector = image_to_feature_vector(image_path)
        feature_vector = feature_vector.reshape(1, -1)
        predicted_label = knn_model.predict(feature_vector)[0]

        # Handle unknown labels
        if 0 <= predicted_label <= 3:
            shape_name = list(label_mapping.keys())[list(label_mapping.values()).index(predicted_label)]
        else:
            shape_name = "other"

        return shape_name
```
```python
# 5. Main Execution
# Load and split data
X, y = load_image_data('/content/drive/MyDrive/Electives_Final/archive/data')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
```
```python
# Train KNN model
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
```# Evaluate the model
y_pred = knn.predict(X_test)
accuracy = metrics.accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Generate a classification report
print(metrics.classification_report(y_test, y_pred))

#Confusion Matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=list(label_mapping.keys()),
            yticklabels=list(label_mapping.keys()))
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()
```
```python
label_mapping = {0: 'triangle', 1: 'rectangle', 2: 'ellipse', 3: 'other'} # Updated for inverse mapping

# Display some images and predictions
fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(10, 4))
for i, ax in enumerate(axes.flat):
    ax.imshow(X_test[i].reshape(70, 70, 4), cmap='gray')

    # Get shape names for labels
    true_label = label_mapping[y_test[i]]
    predicted_label = label_mapping[y_pred[i]]

    # Set title with true above predicted
    ax.set_title(f"True: {true_label}\nPredicted: {predicted_label}")
    ax.axis('off')

plt.show()
```
```python
import os
from google.colab import files
import cv2
import numpy as np
import matplotlib.pyplot as plt

uploaded = files.upload()

for fn in uploaded.keys():
  print('User uploaded file "{name}" with length {length} bytes'.format(
      name=fn, length=len(uploaded[fn])))

  # Predict the shape of the uploaded image
  predicted_shape = predict_shape(fn, knn)  # Assuming 'knn' is your trained model
  print(f"Predicted shape for {fn}: {predicted_shape}")

  # Display the image and draw contours
  img = cv2.imread(fn)
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
  contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

  if contours:
    largest_contour = max(contours, key=cv2.contourArea)
    cv2.drawContours(img, [largest_contour], -1, (0, 255, 0), 3)  # Draw green contour

  plt.figure(figsize=(6, 6))
  plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) # Convert BGR to RGB for matplotlib
  plt.title(f"Predicted Shape: {predicted_shape}")
  plt.axis('off')
  plt.show()
```




### Video Demonstration:
https://drive.google.com/file/d/1B6CUIUTUZzZWW-h_52ndpdoK4DHrArgl/view?usp=sharing

### Dataset Gdrive Link:
https://drive.google.com/drive/folders/1L0N5ZpHeD2r2VjrFlzw7UF3ZAwcl7_zQ?usp=sharing

### Google Collab Link:
https://colab.research.google.com/drive/10YXFSgrA2Rc6ObqFjc9veMKR8LmolHVL?usp=sharing

## Refference:
https://www.kaggle.com/datasets/frobert/handdrawn-shapes-hds-dataset/data
