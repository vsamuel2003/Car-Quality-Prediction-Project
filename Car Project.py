import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn import linear_model, preprocessing, model_selection

COLUMN_NAMES = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'quality']

data = pd.read_csv('car.data', names=COLUMN_NAMES, header=0)

X = data.drop(columns = ['quality'])
y = data['quality']

buying = le.fit_transform(list(data['buying']))
maint = le.fit_transform(list(data['maint']))
doors = le.fit_transform(list(data['doors']))
persons = le.fit_transform(list(data['persons']))
lug_boot = le.fit_transform(list(data['lug_boot']))
safety = le.fit_transform(list(data['safety']))
cls = le.fit_transform(list(data['quality']))

predict = 'quality'

X = list(zip(buying,maint,doors,persons,lug_boot,safety))
y = list(cls)

X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y, test_size=0.1)

n_neighbors = 9
model = KNeighborsClassifier(n_neighbors=n_neighbors)
model.fit(X_train,y_train)
acc = model.score(X_test,y_test)
print(f'Accuracy = {acc}')

names = ['unacc', 'acc', 'good', 'vgood']
predictions = model.predict(X_test)

for x in range(len(predictions)):
    print(f' prediction: {names[predictions[x]]} data: {X_test[x]} actual: {names[y_test[x]]}')