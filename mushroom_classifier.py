import numpy as np

from bokeh.io import curdoc
from bokeh.layouts import widgetbox, row, column
from bokeh.models import ColumnDataSource, Select, Button, HoverTool, Range1d
from bokeh.plotting import figure

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier




np.random.seed(0)



# define some helper functions
def fit_model(X_train, y_train, model_to_be_fitted):

    # Generate the new colors:
    if model_to_be_fitted == 'Logistic Regression':
        model = LogisticRegression()

    elif model_to_be_fitted == 'KNN (N=20)':
        model = KNeighborsClassifier(n_neighbors=20)

    elif model_to_be_fitted == 'KNN (N=1)':
        model = KNeighborsClassifier(n_neighbors=1)

    elif model_to_be_fitted == 'Decision Tree (max_depth=5)':
        model = DecisionTreeClassifier(max_depth=5)

    elif model_to_be_fitted == 'Decision Tree (full depth)':
        model = DecisionTreeClassifier()

    elif model_to_be_fitted == 'Random Forest':
        model = RandomForestClassifier()

    elif model_to_be_fitted == 'SVM (linear kernel)':
        model = SVC(kernel='linear', probability=True)

    elif model_to_be_fitted == 'SVM (rbf kernel)':
        model = SVC(kernel='rbf', C=10, probability=True)

    elif model_to_be_fitted == 'Gaussian Naive Bayes':
        model = GaussianNB()

    elif model_to_be_fitted == 'MLP (3 hidden layers, relu activation)':
        model = MLPClassifier(hidden_layer_sizes=(20,20,20), activation='relu', solver='lbfgs')

    else:
        model = LogisticRegression()

    return model.fit(X_train, y_train)


mesh_step_size = 0.03
x_min, x_max = -2, 2.7
y_min, y_max = -2.4, 2.7


def predictions(X_test, model, prob):
    xx, yy = np.meshgrid(np.arange(x_min, x_max, mesh_step_size), np.arange(y_min, y_max, mesh_step_size))

    if prob:
        Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
        pred = model.predict_proba(X_test)[:, 1]
    else:
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        pred = model.predict(X_test)

    return Z.reshape(xx.shape), pred


# set up initial data
mushroom = np.load('mushroom.npy')
X, y = mushroom[:, :2], mushroom[:, -1]
model = 'Logistic Regression'
test_size = 0.25
prob = True


X_train, X_test, y_train, y_test = train_test_split(X, y)
fitted_model = fit_model(X_train, y_train, model)
back_preds, point_preds = predictions(X_test, fitted_model, prob=prob)

c = ['#d6d7ea', '#d6d8ea', '#d7d9eb', '#d7dbec', '#d7dcec', '#d8dded', '#d8deee', '#d9e0ee',
     '#d9e1ef', '#d9e2ef', '#dae4f0', '#dbe5f1', '#dce6f1', '#dde7f2', '#dee8f2', '#dfe9f3',
     '#e0eaf4', '#e1ebf4', '#e1edf5', '#e2eef5', '#e3eff6', '#e5f0f6', '#e6f1f7', '#e7f1f7',
     '#e8f2f8', '#e9f3f8', '#eaf4f9', '#ebf5f9', '#ecf6fa', '#edf7fa', '#eff8fb', '#f0f8fb',
     '#f1f9fb', '#f2f9fc', '#f3fafc', '#f4fafc', '#f5fbfd', '#f6fbfd', '#f7fcfd', '#f8fcfd',
     '#f9fdfd', '#fafdfc', '#fafdfb', '#fbfdfa', '#fcfef9', '#fcfef7', '#fdfef6', '#fdfef5',
     '#fefff4', '#fffff3', '#fffff2', '#fffef1', '#fffdf0', '#fffdef', '#fffcee', '#fffced',
     '#fffbec', '#fffaeb', '#fffaea', '#fff9e9', '#fff8e8', '#fff7e7', '#fff6e6', '#fff5e5',
     '#fff4e4', '#fff3e3', '#fff2e3', '#fff1e2', '#fff0e1', '#ffefe0', '#feeedf', '#feedde',
     '#feebde', '#feeadd', '#fee9dd', '#fee7dc', '#fde6db', '#fde5db', '#fde3da', '#fde2da',
     '#fce1d9', '#fce0d8', '#fbded8', '#fbddd7', '#fadcd7', '#f9dbd6', '#f9d9d6', '#f8d8d5',
     '#f8d7d4', '#f7d6d4', '#f6d5d4', '#f5d4d4', '#f4d3d4', '#f3d2d4', '#f2d1d4', '#f1d0d4',
     '#f0cfd4', '#efced4', '#eecdd4', '#edccd4']

TOOLS = "hover,save,pan,box_zoom,reset,wheel_zoom"


source2 = ColumnDataSource(data=dict(pc1=X_test[:, 0], pc2=X_test[:, 1],
                                     color=np.vectorize({0.0: '#036D9B', 1.0: '#E62A05'}.get)(y_test),
                                     poisonous=np.vectorize({0.0: 'no', 1.0: 'yes'}.get)(y_test),
                                     pred=point_preds))


x_range = Range1d(x_min, x_max, bounds=(x_min, x_max), min_interval=1.5)
y_range = Range1d(y_min, y_max, bounds=(y_min, y_max), min_interval=1.5)


p2 = figure(title=model + ' ' * 5 + 'Test Accuracy: ' + str(np.round(fitted_model.score(X, y), 3)),
            x_range=x_range, y_range=y_range, tools=TOOLS,
            plot_width=500, plot_height=500)

p2.toolbar.logo = None
p2.axis.minor_tick_line_color = None

p2.image([back_preds], x=x_min, y=y_min, dw=x_max - x_min, dh=y_max - y_min, palette=c)
p2.circle('pc1', 'pc2', color='color', fill_alpha=0.6, size=9, line_alpha=0, source=source2, level='overlay')

p2.select_one(HoverTool).tooltips = [("poisonous", "@poisonous"), ('prediction', '@pred{0.00}')]


# set up widgets
ml_models = ['Logistic Regression', 'KNN (N=1)', 'KNN (N=20)', 'Decision Tree (max_depth=5)',
             'Decision Tree (full depth)', 'Random Forest', 'SVM (linear kernel)', 'SVM (rbf kernel)',
             'Gaussian Naive Bayes', 'MLP (3 hidden layers, relu activation)']

model_select = Select(value='Logistic Regression',
                      title='Select model:',
                      width=200,
                      options=ml_models)

test_split_button = Button(label="New Train/Test Split", width=200)

prediction_types = ['Probability', 'Decision']

prediction_select = Select(value='Probability',
                           title='Select boundary:',
                           width=200,
                           options=prediction_types)

inputs = column(widgetbox(test_split_button, model_select, prediction_select))


def update_predictions(attrname, old, new):
    global X_train, X_test, y_train, y_test

    model = model_select.value
    # test_size = int(test_size_slider.value)
    prob = True if prediction_select.value == 'Probability' else False

    # X_train, X_test, y_train, y_test = train_test_split(X, y)
    fitted_model = fit_model(X_train, y_train, model)

    back_preds, point_preds = predictions(X_test, fitted_model, prob=prob)

    source2.data = dict(pc1=X_test[:, 0], pc2=X_test[:, 1],
                        color=np.vectorize({0.0:'#036D9B', 1.0:'#E62A05'}.get)(y_test),
                        poisonous=np.vectorize({0.0: 'no', 1.0: 'yes'}.get)(y_test),
                        pred=point_preds)

    p2.image([back_preds], x=x_min, y=y_min, dw=x_max - x_min, dh=y_max - y_min, palette=c)

    p2.title.text = model + ' ' * 5 + 'Test Accuracy: ' + str(np.round(fitted_model.score(X, y), 3))


def update_predictions2():
    global X_train, X_test, y_train, y_test

    model = model_select.value
    # test_size = int(test_size_slider.value)
    prob = True if prediction_select.value == 'Probability' else False

    X_train, X_test, y_train, y_test = train_test_split(X, y)
    fitted_model = fit_model(X_train, y_train, model)

    back_preds, point_preds = predictions(X_test, fitted_model, prob=prob)

    source2.data = dict(pc1=X_test[:, 0], pc2=X_test[:, 1],
                        color=np.vectorize({0.0:'#036D9B', 1.0:'#E62A05'}.get)(y_test),
                        poisonous=np.vectorize({0.0: 'no', 1.0: 'yes'}.get)(y_test),
                        pred=point_preds)

    p2.image([back_preds], x=x_min, y=y_min, dw=x_max - x_min, dh=y_max - y_min, palette=c)

    p2.title.text = model + ' ' * 5 + 'Test Accuracy: ' + str(np.round(fitted_model.score(X, y), 3))



model_select.on_change('value', update_predictions)
test_split_button.on_click(update_predictions2)
prediction_select.on_change('value', update_predictions)


curdoc().add_root(row(inputs, p2))
curdoc().title = "Mushrooms"
