
# Future To-do list

- Check out Spacy library https://spacy.io/
- Can add Sklearn with Keras https://keras.io/scikit-learn-api/
```
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense

def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier)
parameters = {'batch_size': [25, 32],
              'epochs': [100, 500],
              'optimizer': ['adam', 'rmsprop']}
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)
grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_

```
- Few other ideas to validate final model

- using KFold CV and show the different folds scores
- Run model with multiple different random states and show the mean / variance of the results(this one would be great since I have a small dataset).
- What about small changes in the dataset will this affect this model?
- Another really cool idea would be to check out the SHAP library. SHAP (SHapley Additive exPlanations) is a unified approach to explain the output of any machine learning model. SHAP connects game theory with local explanations, uniting several previous methods and representing the only possible consistent and locally accurate additive feature attribution method based on expectations (see the SHAP NIPS paper for details). This is where you can visualize your machine learning model's predictions with visuals such as
https://github.com/slundberg/shap
http://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions

- On Docstring :https://stackoverflow.com/questions/19074745/docstrings-vs-comments
- Google Style Python Docstrings https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html
- Best of the Best Practices" (BOBP) guide to developing in Python https://gist.github.com/sloria/7001839