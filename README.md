# Fraud-Detection-with-Autoencoders
This project demonstrates an anomaly detection system using autoencoders, an unsupervised deep learning technique, for fraud detection. The model is trained on a dataset containing normal and fraudulent credit card transactions and showcases its ability to identify fraudulent patterns.
# Fraud Detection with Autoencoders

This project demonstrates an anomaly detection system using autoencoders, an unsupervised deep learning technique, for fraud detection. The model is trained on a dataset containing normal and fraudulent credit card transactions and showcases its ability to identify fraudulent patterns.

## Dataset

The dataset used for this project is the Credit Card Fraud Detection dataset. It contains a mixture of fraudulent and genuine credit card transactions. You can download the dataset from Kaggle using the following link: [Credit Card Fraud Detection Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud). Please download the dataset and place it in the same directory as the code file. Alternatively, provide the appropriate file path when loading the dataset in the code.

## Code

The code is implemented in Python using the TensorFlow and scikit-learn libraries. It includes the following files:

- `fraud_detection.py`: The main Python script that builds and trains the autoencoder-based fraud detection model.
- `fraud_dataset.csv`: The dataset file containing the credit card transactions (not included in this repository).

To run the code, ensure that you have the necessary dependencies installed. You can run the script by executing the following command:

```bash
python fraud_detection.py
```

The code will train the autoencoder, detect anomalies, classify samples as fraud or not fraud, evaluate the model's performance, and generate an ROC curve.

## Results

After running the code, the results will be displayed in the console. The script will output the accuracy of the model and generate an ROC curve plot, showcasing the model's performance in detecting fraud.


