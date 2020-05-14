import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# -- read model object and data
model = pickle.load(open("argo-demo/serialized/dummy_model.pkl", 'rb'))
data = pd.read_csv('argo-demo/lifecycle-management/data/data.csv')
new_data = data.tail(100)
X_test, y_test = new_data[['x', 'y']], new_data['label']

# -- evaluate model object
accuracy_old = model.score(X_test, y_test) 

# -- retrain model object
X_train, X_test, y_train, y_test = train_test_split(data[['x','y']], data['label'], random_state = 1337)
retrained_model = KNeighborsClassifier(n_neighbors = 7).fit(X_train, y_train) 
accuracy_new = retrained_model.score(X_test, y_test) 

# -- if retraining is better, save model object
print(accuracy_old, accuracy_new)
if accuracy_new > accuracy_old:
    pickle.dump(retrained_model, open('argo-demo/serialized/dummy_model.pkl', 'wb'))
    result = 'updated the model object'
else:
    result = 'retraining the model did not yield superior performance'
print(result)
