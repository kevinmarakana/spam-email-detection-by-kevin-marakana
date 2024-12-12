# spam-email-detection-by-kevin-marakana


Dataset link : https://drive.google.com/drive/folders/1BJfUP-VqNXBlSdFmO_-Wo3N9dU2BW9HO?usp=sharing


This project implements a machine learning pipeline to classify email messages as spam or non-spam (ham). The model is built using natural language processing (NLP) techniques and a neural network model implemented with TensorFlow.

## Key Features
- **Dataset Preprocessing**:
  - Balanced the dataset using downsampling.
  - Removed stopwords and punctuations from email messages.
  - Visualized data distribution and generated word clouds for spam and ham messages.
- **Data Tokenization and Encoding**:
  - Tokenized the text data and padded sequences to ensure uniform input size.
  - Encoded target labels (spam/ham) as binary values.
- **Model Architecture**:
  - Embedding layer to capture semantic relationships.
  - LSTM layer for sequential data processing.
  - Dense layers with ReLU and sigmoid activations.
- **Training and Evaluation**:
  - Used EarlyStopping and ReduceLROnPlateau callbacks for efficient training.
  - Achieved competitive accuracy on the test set.
- **Prediction Function**:
  - Implemented a function to classify custom email messages as spam or ham.

## Libraries Used
- **Data Preprocessing**: Pandas, Numpy, Matplotlib, Seaborn, NLTK
- **Text Representation**: TensorFlow/Keras Tokenizer, Pad Sequences
- **Model Training**: TensorFlow/Keras
- **Visualization**: WordCloud, Matplotlib

## Workflow
1. Load and preprocess the dataset.
2. Visualize data distribution and text content.
3. Tokenize and pad text data.
4. Build and train an LSTM-based neural network.
5. Evaluate model performance and visualize training metrics.
6. Predict spam/ham for custom messages.

## Results
- The model achieved high accuracy on the test set.
- Demonstrated robust spam detection using word embeddings and LSTM networks.

## How to Use
1. Clone the repository and ensure the required libraries are installed.
2. Run the script to train the model or load a pretrained model.
3. Use the `predict_message` function to classify email messages.

## Example Usage
```python
message = "Congratulations! You've won a free ticket to the Bahamas. Call now!"
prediction = predict_message(model, tokenizer, message)
print(f'The message is predicted to be: {prediction}')
```

## Conclusion
This project showcases the application of NLP and deep learning for spam email detection. By leveraging LSTM and text preprocessing techniques, the model provides reliable classification of email messages.

