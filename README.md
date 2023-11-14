# Text Classification with Supervised Learning
## Project Overview
This project focuses on text classification using supervised learning techniques. The goal is to classify posts from various Reddit subreddits into predefined categories such as datascience, machinelearning, physics, astrology, and conspiracy. The classification is based on the content of the posts obtained from the Reddit API.

## Project Structure
The project consists of the following components:

### Data Loading:
- The data is obtained from Reddit using the PRAW (Python Reddit API Wrapper) library.
- The subreddits considered for classification are: datascience, machinelearning, physics, astrology, and conspiracy.

### Data Preprocessing:
- The text data is preprocessed by removing non-alphabetic characters, digits, and URLs.
- The data is then split into training and testing sets for model evaluation.

### Feature Extraction:
- Text data is transformed into numerical features using the TF-IDF (Term Frequency-Inverse Document Frequency) vectorizer.
- Dimensionality reduction is performed using Truncated SVD (Singular Value Decomposition) to improve model efficiency.

### Model Selection:

Three models are considered for classification:
- K-Nearest Neighbors (KNN)
- Random Forest
- Logistic Regression with Cross-Validation

### Model Training and Evaluation:
- Each model is trained on the training data and evaluated on the testing data.
- Classification reports are generated for each model, providing insights into precision, recall, and F1-score.

### Results Visualization:

- The project includes visualizations to better understand the results.
- A bar plot displays the number of posts per subreddit to provide an overview of the dataset.
- Confusion matrices are plotted for each model, offering a detailed view of the classification performance.

## Instructions for Running the Project
### Ensure you have the required dependencies installed. You can install them using:
```
pip install praw scikit-learn matplotlib seaborn
```

### Set up your Reddit API credentials by creating a .env file with the following variables:
```
client_id=YOUR_CLIENT_ID
client_secret=YOUR_CLIENT_SECRET
password=YOUR_REDDIT_PASSWORD
user_agent=YOUR_USER_AGENT
user=YOUR_REDDIT_USERNAME
```

### Run the __main__ script from the root.
```
python .
```

### Explore the generated visualizations to understand the model's performance.

## Conclusion
This text classification project provides a comprehensive approach to categorizing Reddit posts into specific topics using machine learning. The chosen models are evaluated, and visualizations offer insights into the distribution of posts and model predictions.

Feel free to explore, modify, and extend this project for your specific use case or dataset. If you have any questions or suggestions, please don't hesitate to reach out.