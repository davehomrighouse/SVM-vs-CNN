# SVM-vs-CNN
# Project Overview

This project compares traditional machine learning and deep learning approaches for multi-class text classification using the Reuters dataset. The objective was to evaluate how model architecture, feature representation, and class imbalance handling impact performance.

# Models Implemented
Support Vector Machine (SVM) with TF-IDF features
Convolutional Neural Networks (CNNs):
  Baseline CNN
  CNN with modified architecture
  CNN with parallel convolution branches
  CNN with pretrained GLoVe embeddings

# Evaluation Metrics
Models were evaluated using:

Accuracy
Precision, Recall, F1 Score (weighted)
Macro F1 Score (to assess performance across all classes equally)
Training/validation loss curves

# Key Results
| Model                    | Accuracy | F1 (Weighted) | F1 (Macro) |
|--------------------------|----------|---------------|------------|
| SVM                      | 0.805    | 0.806         | 0.693      |
| CNN (Parallel Branches)  | 0.793    | 0.790         | 0.635      |
| CNN (Baseline Variants)  | ~0.75    | ~0.74         | 0.57–0.60  |
| CNN (GLoVe)              | 0.658    | 0.667         | 0.443      |

# Key Insights
1. SVM Outperformed CNN Models

The SVM consistently achieved the highest accuracy and macro F1 score, demonstrating strong performance on smaller datasets with sparse TF-IDF features.

2. CNN Architecture Matters

The CNN with parallel convolution branches performed closest to the SVM, indicating that architectural improvements can significantly enhance feature extraction.

3. Limited Impact of Pretrained Embeddings

The CNN model using GLoVe embeddings underperformed compared to other CNN variants, suggesting that fixed pretrained embeddings may not provide benefits in smaller or domain-specific datasets.

4. Class Imbalance Handling is Critical
Full class weighting introduced instability in CNN training
Scaled class weighting improved performance and minority class representation
Macro F1 revealed performance gaps not visible through accuracy alone
Conclusion

Traditional machine learning methods, particularly SVM with TF-IDF, remain highly competitive for text classification tasks with limited data. While CNNs can approach similar performance with appropriate architecture, they are more sensitive to training dynamics and class imbalance.

# Future Work
Evaluate transformer-based models (e.g., BERT, SBERT)
Explore fine-tuning embeddings (trainable=True)
Investigate advanced imbalance handling techniques (e.g., focal loss, data augmentation)
