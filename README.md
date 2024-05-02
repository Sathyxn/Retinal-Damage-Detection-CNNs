# Retinal Damage Detection with Deep Learning

Retina is a vital component of the eye; it plays a crucial role in visual perception. Any damage to the retina can lead to vision impairment and severe ocular diseases. Early detection and accurate diagnosis of retinal conditions is crucial. In the domain of medical image analysis, especially within the scope of retinal imaging, convolutional neural network (CNN)-based deep learning models have exhibited promising results.

![OCT Images](https://i.imgur.com/fSTeZMd.png)


This project involves the detection of retinal conditions such as Diabetic Macular Edema (DME), Choroidal Neovascularization (CNV), and Drusen. We present a comparative study of CNN models for the detection of these retinal damages using Optical Coherence Tomography (OCT) images. The study focuses on evaluating the performance of popular CNN architectures, including:

- **Residual Networks (ResNet)**
- **Dense Convolutional Networks (DenseNet)**
- **Inception Networks**
- **Visual Geometry Group (VGG)**


The comparative analysis includes different depth variants of deep learning models such as ResNet (18, 50, and 101), DenseNet (121, 169, and 201), Inception-V3, Inception-ResNetV2, and VGG (16 and 19). These models are assessed on a dataset containing OCT images of Normal, DME, CNV, and Drusen retinal conditions.

## Objective

The main objective of this study is to explore the impact of depth on the model performance and understand how deeper models demonstrate an enhanced ability to learn complex features and capture intricate details within the retinal data. By exploring these depth variants, we aim to provide insights into the advantages and potential trade-offs associated with increasing the depth of deep learning models for retinal damage detection tasks.

## Methodology

The effectiveness of the models is evaluated using evaluation metrics such as accuracy, precision, recall, and F1 score. Additionally, visualizations such as confusion matrix, training accuracy versus validation accuracy plots, and training loss versus validation loss plots are utilized to provide a better understanding of the model performance.

## Repository Structure

- **data/**: Contains datasets used for training and testing the models.
- **models/**: Contains pre-trained model files (.h5, .hdf5, etc.) that can be used directly for inference.
- **training_history/**: Contains training history files documenting the training metrics and parameters of each model.
- **plots/**: Contains visualization plots generated during training or evaluation.
- **src/**: Contains the source code for the project, including scripts for model training and evaluation.
- **requirements.txt**: Lists Python packages required to run the project.

## Usage

To use this project, follow these steps:

1. Clone the repository to your local machine.
2. Install the required dependencies listed in `requirements.txt`.
3. Use the provided scripts in the `src/` directory to train and evaluate models.

## Acknowledgements

- **Data**: [Mendeley Dataset](https://data.mendeley.com/datasets/rscbjbr9sj/2), [Kaggle Dataset](https://www.kaggle.com/datasets/paultimothymooney/kermany2018)
- **Citation**: [Original Paper](http://www.cell.com/cell/fulltext/S0092-8674(18)30154-5)

Feel free to reach out with any questions, suggestions, or contributions!
