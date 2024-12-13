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
### Video Demonstration:
https://drive.google.com/file/d/1B6CUIUTUZzZWW-h_52ndpdoK4DHrArgl/view?usp=sharing

### Dataset Gdrive Link:
https://drive.google.com/drive/folders/1L0N5ZpHeD2r2VjrFlzw7UF3ZAwcl7_zQ?usp=sharing

### Google Collab Link:
https://colab.research.google.com/drive/10YXFSgrA2Rc6ObqFjc9veMKR8LmolHVL?usp=sharing

## Refference:
https://www.kaggle.com/datasets/frobert/handdrawn-shapes-hds-dataset/data
