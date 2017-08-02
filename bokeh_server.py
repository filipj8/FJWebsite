import numpy as np
np.random.seed(0)

from bokeh.io import curdoc
from bokeh.layouts import widgetbox, row, column
from bokeh.models import ColumnDataSource, Select, Button, HoverTool, Range1d
from bokeh.plotting import figure

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier



# define some helper functions
def fit_model(X_train, y_train, algorithm):

    # Generate the new colors:
    if algorithm == 'Logistic Regression':
        model = LogisticRegression()

    elif algorithm == 'Decision Tree':
        model = DecisionTreeClassifier(max_depth=3)

    elif algorithm == 'KNN (N=20)':
        model = KNeighborsClassifier(n_neighbors=20)

    else:
        model = LogisticRegression()

    return model.fit(X_train, y_train)


mesh_step_size = 0.03
x_min, x_max = -2, 2.6
y_min, y_max = -2.4, 2.6


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
algorithm = 'Logistic Regression'
test_size = 0.25
prob = True


X_train, X_test, y_train, y_test = train_test_split(X, y)
fitted_model = fit_model(X_train, y_train, algorithm)
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


source = ColumnDataSource(data=dict(pc1=X_test[:, 0], pc2=X_test[:, 1],
                                    color=np.vectorize({0.0: '#036D9B', 1.0: '#E62A05'}.get)(y_test),
                                    poisonous=np.vectorize({0.0: 'no', 1.0: 'yes'}.get)(y_test),
                                    pred=point_preds))


x_range = Range1d(x_min, x_max, bounds=(x_min, x_max), min_interval=1.5)
y_range = Range1d(y_min, y_max, bounds=(y_min, y_max), min_interval=1.5)


p = figure(title=algorithm + ' '*5 + 'Test Accuracy: ' + str(np.round(fitted_model.score(X, y), 3)),
           x_range=x_range, y_range=y_range, tools=TOOLS,
           plot_width=500, plot_height=500)

p.xaxis.minor_tick_line_color = None
p.yaxis.minor_tick_line_color = None

p.image([back_preds], x=x_min, y=y_min, dw=x_max - x_min, dh=y_max - y_min, palette=c)

p.circle('pc1', 'pc2', color='color', fill_alpha=0.6, size=9, line_alpha=0, source=source, level='overlay')

p.select_one(HoverTool).tooltips = [
    ("poisonous", "@poisonous"),
    ('prediction', '@pred{0.00}')]


# set up widgets
ml_algorithms = ['Logistic Regression', 'Decision Tree', 'KNN (N=20)']

algorithm_select = Select(value='Logistic Regression',
                          title='Select algorithm:',
                          width=200,
                          options=ml_algorithms)

test_split_button = Button(label="New Train/Test Split", width=200)

prediction_types = ['Probability', 'Decision']

prediction_select = Select(value='Probability',
                           title='Select boundary:',
                           width=200,
                           options=prediction_types)


# set up callbacks
def update_predictions(attrname, old, new):
    global X_train, X_test, y_train, y_test

    algorithm = algorithm_select.value
    # test_size = int(test_size_slider.value)
    prob = True if prediction_select.value == 'Probability' else False

    # X_train, X_test, y_train, y_test = train_test_split(X, y)
    fitted_model = fit_model(X_train, y_train, algorithm)

    back_preds, point_preds = predictions(X_test, fitted_model, prob=prob)

    source.data = dict(pc1=X_test[:, 0], pc2=X_test[:, 1],
                       color=np.vectorize({0.0:'#036D9B', 1.0:'#E62A05'}.get)(y_test),
                       poisonous=np.vectorize({0.0: 'no', 1.0: 'yes'}.get)(y_test),
                       pred=point_preds)

    p.image([back_preds], x=x_min, y=y_min, dw=x_max - x_min, dh=y_max - y_min, palette=c)

    p.title.text = algorithm + ' '*5 + 'Test Accuracy: ' + str(np.round(fitted_model.score(X, y), 3))

def update_predictions2():
    global X_train, X_test, y_train, y_test

    algorithm = algorithm_select.value
    # test_size = int(test_size_slider.value)
    prob = True if prediction_select.value == 'Probability' else False

    X_train, X_test, y_train, y_test = train_test_split(X, y)
    fitted_model = fit_model(X_train, y_train, algorithm)

    back_preds, point_preds = predictions(X_test, fitted_model, prob=prob)

    source.data = dict(pc1=X_test[:, 0], pc2=X_test[:, 1],
                       color=np.vectorize({0.0:'#036D9B', 1.0:'#E62A05'}.get)(y_test),
                       poisonous=np.vectorize({0.0: 'no', 1.0: 'yes'}.get)(y_test),
                       pred=point_preds)

    p.image([back_preds], x=x_min, y=y_min, dw=x_max - x_min, dh=y_max - y_min, palette=c)

    p.title.text = algorithm + ' '*5 + 'Test Accuracy: ' + str(np.round(fitted_model.score(X, y), 3))


algorithm_select.on_change('value', update_predictions)
test_split_button.on_click(update_predictions2)
prediction_select.on_change('value', update_predictions)

# set up layout
# selects = row(prediction_select, algorithm_select, width=420)
inputs = column(widgetbox(test_split_button, algorithm_select, prediction_select))

# add to document
curdoc().add_root(row(inputs, p))
curdoc().title = "Mushrooms"
