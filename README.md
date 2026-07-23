## 🛢️ Oil & Gas Onshore/Offshore Classification System

A machine learning web application built with Streamlit that classifies oil and gas fields as ONSHORE or OFFSHORE using geological and geographic information. The project includes data preprocessing, model training, evaluation, feature importance analysis, and interactive prediction interfaces for both single-record and batch predictions.

## 🚀 Project Overview

This project demonstrates an end-to-end machine learning workflow for the oil and gas domain:

Data cleaning and preprocessing
Handling missing values with SimpleImputer
Categorical encoding with OneHotEncoder
Model training with RandomForestClassifier
Performance evaluation (Accuracy, Precision, Recall, F1-score)
Cross-validation
Feature importance visualization
Interactive Streamlit dashboard
Single-field prediction
Batch prediction from uploaded CSV files

The application is designed to be production-ready by using a scikit-learn Pipeline, ensuring that the same preprocessing steps are applied consistently during training and inference.