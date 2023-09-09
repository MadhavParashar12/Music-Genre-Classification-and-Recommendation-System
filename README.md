# Music-Genre-Classification
### Introduction
The Music Genre Classification and Recommendation System is an innovative machine-learning project aimed at classifying music genres and providing personalized song recommendations based on a user's preferences. This project leverages the power of the k-Nearest Neighbors (KNN) algorithm for genre classification, utilizes the librosa library for feature extraction, employs the Random Forest algorithm to select important features, and recommends songs from a self-created CSV dataset.
### Project Goals
1. Music Genre Classification: The primary goal of this project is to accurately classify music tracks into predefined genres. By training a KNN classifier on a dataset of labeled music samples, we aim to develop a model capable of automatically categorizing songs into genres such as rock, jazz, hip-hop, classical, and more.

2. Feature Extraction: To effectively classify music genres, we will use the librosa library to extract relevant audio features from music tracks. These features may include but are not limited to tempo, spectral centroid, chroma feature, and mel-frequency cepstral coefficients (MFCCs).

3. Feature Selection: We will employ the Random Forest algorithm to identify and select the most important features for music genre classification. This step will help improve the model's accuracy by focusing on the most discriminative audio characteristics.

4. Song Recommendation: Once a user provides input regarding their musical preferences, the system will classify their favorite genre(s). Based on this classification, the system will recommend songs from a curated CSV file containing a collection of songs and their associated genres.
