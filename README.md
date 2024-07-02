# Sentiment_Analysis-of-data
The project deals with  Sentiment Analysis of flipkart website .
The working is done in following way 
Step 1:
Importing Libraries and Loading the Dataset
Libraries Used:

Pandas: For data manipulation and analysis.
Seaborn and Matplotlib: For visualization.
NLTK: For natural language processing.
TfidfVectorizer: For converting text into numerical vectors.



Step 2:
Data Exploration
Unique Ratings:

Identify the unique ratings present in the dataset to understand the rating distribution.
Countplot:

Visualize the count of each rating using a count plot.



Step 3:
Label Creation for Sentiment Analysis
Binary Sentiment Labels:

Convert ratings into binary sentiment labels:
1: Positive sentiment (rating of 5).
0: Negative sentiment (ratings less than 5).
This helps in simplifying the problem to a binary classification task.


Step 4:
 Data Preprocessing
Text Preprocessing:

Remove punctuations using regular expressions.
Convert text to lowercase and remove stopwords (common words that don’t contribute much to the meaning, e.g., "the", "is").
Tokenize the text (break down into individual words).





Step 5:
Dataset Analysis
Label Count:

Count the number of positive and negative labels to understand the balance of the dataset.
Word Cloud:

Generate a word cloud for positive reviews to visualize the most frequent words. This helps in understanding which words are commonly associated with positive sentiments.


Step 6:

Text Vectorization
TF-IDF Vectorization:

Convert the preprocessed text into numerical vectors using TF-IDF (Term Frequency-Inverse Document Frequency).
TF-IDF measures the importance of a word in a document relative to a collection of documents.
This step transforms textual data into a format that machine learning models can work with.444



Step7:

Model Training, Evaluation, and Prediction
Train-Test Split:

Split the data into training and testing sets (67% training, 33% testing) while maintaining the same distribution of labels in both sets.
Model Training:

Train a Decision Tree Classifier on the training data.
A Decision Tree Classifier is a simple and intuitive model that splits the data based on feature values to make predictions.
Model Evaluation:

Predict the labels for the training data and calculate the accuracy.
Display a confusion matrix to visualize the performance of the model (true positives, false positives, true negatives, and false negatives).



Step8:


 Conclusion and Future Work
The Decision Tree Classifier shows good performance on the training data.
Future work could involve using larger datasets and experimenting with other machine learning algorithms (e.g., Random Forest, Support Vector Machine, Neural Networks) for potentially better performance.



The project demonstrates the complete pipeline for text-based sentiment analysis using machine learning.
Preprocessing and feature extraction are crucial steps for converting text data into a format suitable for machine learning.
Visualization helps in understanding the data and model performance.
The model can be further improved by using more advanced algorithms and larger datasets.
