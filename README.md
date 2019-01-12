# Sentiment analysis of Amazon 500,000 fine food review

- In this project I look at the review helpfulness and interaction between top reviewers in the Amazon community.
- Study the impact of time on the review score
- Perform NLP analysis on the review sentiments to categorize positive and negative reviews
- Analyze word structures and relation using unsupervised methods, and compare the unsupervised results with supervised classification


# Dependecies and packages:

- Python 3.x
- Numpy
- Pandas
- Scikit learn
- Keras (Tensor flow backend)
- tqdm
- seaborn
- matplotlib
- bokeh
- IPython
- itertools
- collections

# Repository layout

├── amazon_finefood_review_project1.ipynb
├── images
├── LICENSE
├── README.md
└── stopword_short_long_mod.txt


# Repository content:

- Jupter notebook file: `amazon_finefood_review_project1.ipynb`
- Image folder (`images`): containing figures in the notebook
- A txt file : `stopword_short_long_mod.txt`, containg list of stop words
- MIT License file

# How to run the code:

Clone the repository (git clone) using the Github address: `https://github.com/ania4data/Sentiment_study_Amazon_fine_food_review.git`


# Source dataset

The zipped `csv` file used in this study can be downloaded from:

- [Web data: Amazon Fine Foods reviews](https://snap.stanford.edu/data/web-FineFoods.html)
- [Kaggle](https://www.kaggle.com/snap/amazon-fine-food-reviews)


# Link to the blog post

[Here is a blog post on the study on medium](https://medium.com/@anoosheh.niavarani.egr/what-an-amazon-fine-food-review-tell-us-a-food-for-thought-869dfe70f2ee)


# Main conclusions

- The analysis reveal two group of reviewers stand out in helpfulness to the community: First group are those who write long reviews, and second are the ones writing very frequent shorter reviews.
- The increase in the Amazon 5-start reviews after 2006 is alarming, that may indicate sign of abuse of the system. - In the past few years Amazon has introduced verified accounts, put more restriction on who is considered verified customers as well as introducing vine program. It would be interesting to see if these changes impact review score trend.
- The machine learning algorithm can successfully predict word sentiment of reviews that the model has not seen before with score ROC_AUC=0.95, Accuracy and F1_score of 0.9.
- The word structure created from embedding vectors after training the model clearly show two blob structure by unsupervised t-SNE representing sentiments (positive and negative) in the data.
- Considering various deep-learning approch tested in the process of completing this project (e.g. GRU, LSTM, 1D-Conv) the results did not improve significantly with adding deeper layer or changinf drop-out, due to alarming increase in 5-star reviews in Amazon data one approach is trying to screen these "fake" reviews by having information about the review submitter. This will hopefully improve the unbalanced nature of the reviews and perhaps the overall performance of the model.