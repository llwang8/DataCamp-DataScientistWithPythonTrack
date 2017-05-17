# DataCamp
# Machine Learning With Experts: School Budgets




# EDA

# Print the summary statistics
print(df.describe())
# Import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
# Create the histogram
plt.hist(df['FTE'].dropna())
# Add title and labels
plt.title('Distribution of %full-time \n employee works')
plt.xlabel('% of full-time')
plt.ylabel('num employees')
# Display the histogram
plt.show()


# Encode the labels as categorical variables

# Define the lambda function: categorize_label
categorize_label = lambda x: x.astype('category')
# Convert df[LABELS] to a categorical type
df[LABELS] = df[LABELS].apply(categorize_label, axis=0)
# Print the converted dtypes
print(df[LABELS].dtypes)



# Counting unique labels

# Import matplotlib.pyplot
import matplotlib.pyplot as plt
# Calculate number of unique values for each label: num_unique_labels
num_unique_labels = df[LABELS].apply(pd.Series.nunique)
# Plot number of unique values for each label
num_unique_labels.plot(kind='bar')
# Label the axes
plt.xlabel('Labels')
plt.ylabel('Number of unique values')
# Display the plot
plt.show()



# Penalizing highly confident wrong answer
# As Peter explained in the video, log loss provides a steep penalty for
# predictions that are both wrong and confident, i.e., a high probability
# is assigned to the incorrect class.


# Compute and print log loss for 1st case
correct_confident = compute_log_loss(correct_confident, actual_labels)
print("Log loss, correct and confident: {}".format(correct_confident))
# Compute log loss for 2nd case
correct_not_confident = compute_log_loss(correct_not_confident, actual_labels)
print("Log loss, correct and not confident: {}".format(correct_not_confident))
# Compute and print log loss for 3rd case
wrong_not_confident = compute_log_loss(wrong_not_confident, actual_labels)
print("Log loss, wrong and not confident: {}".format(wrong_not_confident))
# Compute and print log loss for 4th case
wrong_confident = compute_log_loss(wrong_confident, actual_labels)
print("Log loss, wrong and confident: {}".format(wrong_confident))
# Compute and print log loss for actual labels
actual_labels = compute_log_loss(actual_labels, actual_labels)
print("Log loss, actual labels: {}".format(actual_labels))

# <script.py> output:
#    Log loss, correct and confident: 0.05129329438755058
#    Log loss, correct and not confident: 0.4307829160924542
#    Log loss, wrong and not confident: 1.049822124498678
#    Log loss, wrong and confident: 2.9957322735539904
#    Log loss, actual labels: 9.99200722162646e-15


# Creating a simple model

# Setting up a train-test split in scikit-learn
def multilabel_train_test_split(X, Y, size, min_count=5, seed=None):
    """ Takes a features matrix `X` and a label matrix `Y` and
        returns (X_train, X_test, Y_train, Y_test) where all
        classes in Y are represented at least `min_count` times.
    """
    index = Y.index if isinstance(Y, pd.DataFrame) else np.arange(Y.shape[0])
    test_set_idxs = multilabel_sample(Y, size=size, min_count=min_count, seed=seed)
    train_set_idxs = np.setdiff1d(index, test_set_idxs)
    test_set_mask = index.isin(test_set_idxs)
    train_set_mask = ~test_set_mask
    return (X[train_set_mask], X[test_set_mask], Y[train_set_mask], Y[test_set_mask])


# Training a model
# Import classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
# Create the DataFrame: numeric_data_only
numeric_data_only = df[NUMERIC_COLUMNS].fillna(-1000)
# Get labels and convert to dummy variables: label_dummies
label_dummies = pd.get_dummies(df[LABELS])
# Create training and test sets
X_train, X_test, y_train, y_test = multilabel_train_test_split(numeric_data_only,
                                                               label_dummies,
                                                               size=0.2,
                                                               seed=123)
# Instantiate the classifier: clf
clf = OneVsRestClassifier(LogisticRegression())
# Fit the classifier to the training data
clf.fit(X_train, y_train)
# Print the accuracy
print("Accuracy: {}".format(clf.score(X_test, y_test)))


# Instantiate the classifier: clf
clf = OneVsRestClassifier(LogisticRegression())
# Fit it to the training data
clf.fit(X_train, y_train)
# Load the holdout data: holdout
holdout = pd.read_csv('HoldoutData.csv', index_col=0)
# Generate predictions: predictions
predictions = clf.predict_proba(holdout[NUMERIC_COLUMNS].fillna(-1000))


# Instantiate the classifier: clf
clf = OneVsRestClassifier(LogisticRegression())
# Fit it to the training data
clf.fit(X_train, y_train)
# Load the holdout data: holdout
holdout = pd.read_csv('HoldoutData.csv', index_col=0)
# Generate predictions: predictions
predictions = clf.predict_proba(holdout[NUMERIC_COLUMNS].fillna(-1000))



# Generate predictions: predictions
predictions = clf.predict_proba(holdout[NUMERIC_COLUMNS].fillna(-1000))

# Format predictions in DataFrame: prediction_df
prediction_df = pd.DataFrame(columns=pd.get_dummies(df[LABELS]).columns,
                             index=holdout.index,
                             data=predictions)
# Save prediction_df to csv
prediction_df.to_csv('predictions.csv')
# Submit the predictions for scoring: score
score = score_submission(pred_path='predictions.csv')
# Print score
print('Your model, trained with numeric data only, yields logloss score: {}'.format(score))
# <script.py> output:
#    Your model, trained with numeric data only, yields logloss score: 1.9067227623381413



# creating a bag-of-words in scikit-learn
# Import CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer
# Create the token pattern: TOKENS_ALPHANUMERIC
TOKENS_ALPHANUMERIC = '[A-Za-z0-9]+(?=\\s+)'
# Fill missing values in df.Position_Extra
df.Position_Extra.fillna('', inplace=True)
# Instantiate the CountVectorizer: vec_alphanumeric
vec_alphanumeric = CountVectorizer(token_pattern=TOKENS_ALPHANUMERIC)
# Fit to the data
vec_alphanumeric.fit(df.Position_Extra)
# Print the number of tokens and first 15 tokens
msg = "There are {} tokens in Position_Extra if we split on non-alpha numeric"
print(msg.format(len(vec_alphanumeric.get_feature_names())))
print(vec_alphanumeric.get_feature_names()[:15])


# Define combine_text_columns()
def combine_text_columns(data_frame, to_drop=NUMERIC_COLUMNS + LABELS):
    """ converts all text in each row of data_frame to single vector """
    # Drop non-text columns that are in the df
    to_drop = set(to_drop) & set(data_frame.columns.tolist())
    text_data = data_frame.drop(to_drop, axis=1)
    # Replace nans with blanks
    text_data.fillna("", inplace=True)
    # Join all text items in a row that have a space in between
    return text_data.apply(lambda x: " ".join(x), axis=1)



# Import the CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer
# Create the basic token pattern
TOKENS_BASIC = '\\S+(?=\\s+)'
# Create the alphanumeric token pattern
TOKENS_ALPHANUMERIC = '[A-Za-z0-9]+(?=\\s+)'
# Instantiate basic CountVectorizer: vec_basic
vec_basic = CountVectorizer(token_pattern=TOKENS_BASIC)
# Instantiate alphanumeric CountVectorizer: vec_alphanumeric
vec_alphanumeric = CountVectorizer(token_pattern=TOKENS_ALPHANUMERIC)
# Create the text vector
text_vector = combine_text_columns(df)
# Fit and transform vec_basic
vec_basic.fit_transform(text_vector)
# Print number of tokens of vec_basic
print("There are {} tokens in the dataset".format(len(vec_basic.get_feature_names())))
# Fit and transform vec_alphanumeric
vec_alphanumeric.fit_transform(text_vector)
# Print number of tokens of vec_alphanumeric
print("There are {} alpha-numeric tokens in the dataset".format(len(vec_alphanumeric.get_feature_names())))
# <script.py> output:
#    There are 1405 tokens in the dataset
#    There are 1117 alpha-numeric tokens in the dataset
# Notice that tokenizing on alpha-numeric tokens reduced the number of tokens.



# Improving your model

# Instantiate pipeline
# Import Pipeline
from sklearn.pipeline import Pipeline
# Import other necessary modules
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
# Split and select numeric data only, no nans
X_train, X_test, y_train, y_test = train_test_split(sample_df[['numeric']],
                                                    pd.get_dummies(sample_df['label']),
                                                    random_state=22)
# Instantiate Pipeline object: pl
pl = Pipeline([
        ('clf', OneVsRestClassifier(LogisticRegression()))
    ])
# Fit the pipeline to the training data
pl.fit(X_train, y_train)
# Compute and print accuracy
accuracy = pl.score(X_test, y_test)
print("\nAccuracy on sample data - numeric, no nans: ", accuracy)
# <script.py> output:
#    Accuracy on sample data - numeric, no nans:  0.62


# Preprocessing numeric features
# Import the Imputer object
from sklearn.preprocessing import Imputer
# Create training and test sets using only numeric data
X_train, X_test, y_train, y_test = train_test_split(sample_df[['numeric', 'with_missing']], pd.get_dummies(sample_df['label']), random_state=456)
# Insantiate Pipeline object: pl
pl = Pipeline([
        ('imp', Imputer()),
        ('clf', OneVsRestClassifier(LogisticRegression()))
    ])
# Fit the pipeline to the training data
pl.fit(X_train, y_train)
# Compute and print accuracy
accuracy = pl.score(X_test, y_test)
print("\nAccuracy on sample data - all numeric, incl nans: ", accuracy)


# Test Features and Features Union

# Preprocessing text features
# Import the CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer
# Split out only the text data
X_train, X_test, y_train, y_test = train_test_split(sample_df['text'],
                    pd.get_dummies(sample_df['label']), random_state=456)
# Instantiate Pipeline object: pl
pl = Pipeline([
        ('vec', CountVectorizer()),
        ('clf', OneVsRestClassifier(LogisticRegression()))
    ])
# Fit to the training data
pl.fit(X_train, y_train)
# Compute and print accuracy
accuracy = pl.score(X_test, y_test)
print("\nAccuracy on sample data - just text data: ", accuracy)
#<script.py> output:
#    Accuracy on sample data - just text data:  0.808



# Multiple types of processing: Function Transformer
# Import FunctionTransformer
from sklearn.preprocessing import FunctionTransformer
# Obtain the text data: get_text_data
get_text_data = FunctionTransformer(lambda x: x['text'], validate=False)
# Obtain the numeric data: get_numeric_data
get_numeric_data = FunctionTransformer(lambda x: x[['numeric', 'with_missing']], validate=False)
# Fit and transform the text data: just_text_data
just_text_data = get_text_data.fit_transform(sample_df)
# Fit and transform the numeric data: just_numeric_data
just_numeric_data = get_numeric_data.fit_transform(sample_df)
# Print head to check results
print('Text Data')
print(just_text_data.head())
print('\nNumeric Data')
print(just_numeric_data.head())


# Multiple types of processing: FeatureUnion
# Import FeatureUnion
from sklearn.pipeline import FeatureUnion

# Split using ALL data in sample_df
X_train, X_test, y_train, y_test = train_test_split(sample_df[['numeric', 'with_missing', 'text']], pd.get_dummies(sample_df['label']), random_state=22)

# Create a FeatureUnion with nested pipeline: process_and_join_features
process_and_join_features = FeatureUnion(
            transformer_list = [
                ('numeric_features', Pipeline([
                    ('selector', get_numeric_data),
                    ('imputer', Imputer())
                ])),
                ('text_features', Pipeline([
                    ('selector', get_text_data),
                    ('vectorizer', CountVectorizer())
                ]))
             ]
        )
# Instantiate nested pipeline: pl
pl = Pipeline([
        ('union', process_and_join_features),
        ('clf', OneVsRestClassifier(LogisticRegression()))
    ])
# Fit pl to the training data
pl.fit(X_train, y_train)
# Compute and print accuracy
accuracy = pl.score(X_test, y_test)
print("\nAccuracy on sample data - all data: ", accuracy)
# <script.py> output:
#    Accuracy on sample data - all data:  0.928



# Using FunctionTransformer on the main dataset
# Import FunctionTransformer
from sklearn.preprocessing import FunctionTransformer
# Get the dummy encoding of the labels
dummy_labels = pd.get_dummies(df[LABELS])
# Get the columns that are features in the original df
NON_LABELS = [c for c in df.columns if c not in LABELS]
# Split into training and test sets
X_train, X_test, y_train, y_test = multilabel_train_test_split(df[NON_LABELS],
                                                               dummy_labels,
                                                               0.2,                                                             seed=123)
# Preprocess the text data: get_text_data
get_text_data = FunctionTransformer(combine_text_columns, validate=False)
# Preprocess the numeric data: get_numeric_data
get_numeric_data = FunctionTransformer(lambda x: x[NUMERIC_COLUMNS], validate=False)

# Add a model to the pipeline
# Complete the pipeline: pl
pl = Pipeline([
        ('union', FeatureUnion(
            transformer_list = [
                ('numeric_features', Pipeline([
                    ('selector', get_numeric_data),
                    ('imputer', Imputer())
                ])),
                ('text_features', Pipeline([
                    ('selector', get_text_data),
                    ('vectorizer', CountVectorizer())
                ]))
             ]
        )),
        ('clf', OneVsRestClassifier(LogisticRegression()))
    ])
# Fit to the training data
pl.fit(X_train, y_train)
# Compute and print accuracy
accuracy = pl.score(X_test, y_test)
print("\nAccuracy on budget dataset: ", accuracy)
# Accuracy on budget dataset:  0.203846153846


# Try a different class of model
# Import random forest classifer
from sklearn.ensemble import RandomForestClassifier
# Edit model step in pipeline
pl = Pipeline([
        ('union', FeatureUnion(
            transformer_list = [
                ('numeric_features', Pipeline([
                    ('selector', get_numeric_data),
                    ('imputer', Imputer())
                ])),
                ('text_features', Pipeline([
                    ('selector', get_text_data),
                    ('vectorizer', CountVectorizer())
                ]))
             ]
        )),
        ('clf', RandomForestClassifier())
    ])
# Fit to the training data
pl.fit(X_train, y_train)
# Compute and print accuracy
accuracy = pl.score(X_test, y_test)
print("\nAccuracy on budget dataset: ", accuracy)
# Accuracy on budget dataset:  0.301923076923



# Can you adjust the model or parameter to improve accuracy
# Import RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
# Add model step to pipeline: pl
pl = Pipeline([
        ('union', FeatureUnion(
            transformer_list = [
                ('numeric_features', Pipeline([
                    ('selector', get_numeric_data),
                    ('imputer', Imputer())
                ])),
                ('text_features', Pipeline([
                    ('selector', get_text_data),
                    ('vectorizer', CountVectorizer())
                ]))
             ]
        )),
        ('clf', RandomForestClassifier(n_estimators=15))
    ])
# Fit to the training data
pl.fit(X_train, y_train)
# Compute and print accuracy
accuracy = pl.score(X_test, y_test)
print("\nAccuracy on budget dataset: ", accuracy)
# Accuracy on budget dataset:  0.328846153846

# Learning from the expert

# Deciding what's a word
# Import the CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer
# Create the text vector
text_vector = combine_text_columns(X_train)
# Create the token pattern: TOKENS_ALPHANUMERIC
TOKENS_ALPHANUMERIC = '[A-Za-z0-9]+(?=\\s+)'
# Instantiate the CountVectorizer: text_features
text_features = CountVectorizer(token_pattern=TOKENS_ALPHANUMERIC)
# Fit text_features to the text vector
text_features.fit(text_vector)
# Print the first 10 tokens
print(text_features.get_feature_names()[:10])


# N-gram range in scikit-learn

# Import pipeline
from sklearn.pipeline import Pipeline
# Import classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
# Import CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer
# Import other preprocessing modules
from sklearn.preprocessing import Imputer
from sklearn.feature_selection import chi2, SelectKBest
# Select 300 best features
chi_k = 300
# Import functional utilities
from sklearn.preprocessing import FunctionTransformer, MaxAbsScaler
from sklearn.pipeline import FeatureUnion
# Perform preprocessing
get_text_data = FunctionTransformer(combine_text_columns, validate=False)
get_numeric_data = FunctionTransformer(lambda x: x[NUMERIC_COLUMNS], validate=False)
# Create the token pattern: TOKENS_ALPHANUMERIC
TOKENS_ALPHANUMERIC = '[A-Za-z0-9]+(?=\\s+)'
# Instantiate pipeline: pl
pl = Pipeline([
        ('union', FeatureUnion(
            transformer_list = [
                ('numeric_features', Pipeline([
                    ('selector', get_numeric_data),
                    ('imputer', Imputer())
                ])),
                ('text_features', Pipeline([
                    ('selector', get_text_data),
                    ('vectorizer', CountVectorizer(token_pattern=TOKENS_ALPHANUMERIC,
                                                   ngram_range=(1, 2))),
                    ('dim_red', SelectKBest(chi2, chi_k))
                ]))
             ]
        )),
        ('scale', MaxAbsScaler()),
        ('clf', OneVsRestClassifier(LogisticRegression()))
    ])
# Log loss score: 1.2681



# Implement interaction model in scikit-learn
# Instantiate pipeline: pl
pl = Pipeline([
        ('union', FeatureUnion(
            transformer_list = [
                ('numeric_features', Pipeline([
                    ('selector', get_numeric_data),
                    ('imputer', Imputer())
                ])),
                ('text_features', Pipeline([
                    ('selector', get_text_data),
                    ('vectorizer', CountVectorizer(token_pattern=TOKENS_ALPHANUMERIC,
                                                   ngram_range=(1, 2))),
                    ('dim_red', SelectKBest(chi2, chi_k))
                ]))
             ]
        )),
        ('int', SparseInteractions(degree=2)),
        ('scale', MaxAbsScaler()),
        ('clf', OneVsRestClassifier(LogisticRegression()))
    ])
#Log loss score: 1.2256



# Implementing the hashing trick in scikit-learn
# Import HashingVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
# Get text data: text_data
text_data = combine_text_columns(X_train)
# Create the token pattern: TOKENS_ALPHANUMERIC
TOKENS_ALPHANUMERIC = '[A-Za-z0-9]+(?=\\s+)'
# Instantiate the HashingVectorizer: hashing_vec
hashing_vec = HashingVectorizer(token_pattern=TOKENS_ALPHANUMERIC)
# Fit and transform the Hashing Vectorizer
hashed_text = hashing_vec.fit_transform(text_data)
# Create DataFrame and print the head
hashed_df = pd.DataFrame(hashed_text.data)
print(hashed_df.head())
# <script.py> output:
#              0
#    0 -0.160128
#    1  0.160128
#    2 -0.480384
#    3 -0.320256
#    4  0.160128



# Build the winning model

# Import the hashing vectorizer
from sklearn.feature_extraction.text import HashingVectorizer
# Instantiate the winning model pipeline: pl
pl = Pipeline([
        ('union', FeatureUnion(
            transformer_list = [
                ('numeric_features', Pipeline([
                    ('selector', get_numeric_data),
                    ('imputer', Imputer())
                ])),
                ('text_features', Pipeline([
                    ('selector', get_text_data),
                    ('vectorizer', HashingVectorizer(token_pattern=TOKENS_ALPHANUMERIC,
                                                     non_negative=True, norm=None, binary=False,
                                                     ngram_range=(1, 2))),
                    ('dim_red', SelectKBest(chi2, chi_k))
                ]))
             ]
        )),
        ('int', SparseInteractions(degree=2)),
        ('scale', MaxAbsScaler()),
        ('clf', OneVsRestClassifier(LogisticRegression()))
    ])
# Log loss: 1.2258.
# The winner used skillful NLP, efficient computation, and simple but powerful
# stats tricks to master the budget data.






