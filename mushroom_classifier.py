import numpy as np

from bokeh.io import curdoc
from bokeh.layouts import widgetbox, row, column
from bokeh.models import ColumnDataSource, Select, Button, HoverTool, Range1d
from bokeh.plotting import figure

from matplotlib.cm import RdYlBu_r
from PIL import Image

import pickle
import gzip

import itertools


np.random.seed(0)


with gzip.open('model_data.pklz', 'rb') as f:
    model_dict = pickle.load(f)


x_min, x_max = -2, 2.9
y_min, y_max = -2.4, 2.9


# set up initial data
model = 'Logistic Regression'
states = itertools.cycle(range(5))
state = next(states)

X_train = model_dict[model][state]['X_train']
X_test = model_dict[model][state]['X_test']
y_train = model_dict[model][state]['y_train']
y_test = model_dict[model][state]['y_test']

back_preds = model_dict[model][state]['Z_prob']
point_preds = model_dict[model][state]['pred']


TOOLS = "hover,save,pan,box_zoom,reset,wheel_zoom"


source2 = ColumnDataSource(data=dict(pc1=X_test[:, 0], pc2=X_test[:, 1],
                                     color=np.vectorize({0.0: '#036D9B', 1.0: '#E62A05'}.get)(y_test),
                                     poisonous=np.vectorize({0.0: 'no', 1.0: 'yes'}.get)(y_test),
                                     pred=point_preds))


x_range = Range1d(x_min, x_max)
y_range = Range1d(y_min, y_max)


p2 = figure(title=model + ' ' * 5 + 'Test Accuracy: ' + str(np.round(model_dict[model][state]['score'], 3)),
            x_range=x_range, y_range=y_range, tools=TOOLS,
            plot_width=600, plot_height=600, active_scroll='wheel_zoom')

p2.xgrid.grid_line_color = None
p2.ygrid.grid_line_color = None

p2.toolbar.logo = None
p2.axis.minor_tick_line_color = None

background = Image.fromarray(np.uint8(RdYlBu_r(back_preds) * 255))
background.putalpha(30)
background = background.transpose(Image.FLIP_TOP_BOTTOM)
# background = background.resize((164, 177), Image.ANTIALIAS)
white = Image.new('RGBA', background.size, (255, 255, 255))
alpha_composite = Image.alpha_composite(white, background)
alpha_composite.convert('RGB')
xdim, ydim = alpha_composite.size
img = np.empty((ydim, xdim), dtype=np.uint32)
view = img.view(dtype=np.uint8).reshape((ydim, xdim, 4))
view[:, :, :] = np.flipud(np.asarray(alpha_composite))


p2.image_rgba(image=[img], x=-3, y=-3, dw=7, dh=7)
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
    prob = 'Z_prob' if prediction_select.value == 'Probability' else 'Z'

    back_preds = model_dict[model][state][prob]
    point_preds = model_dict[model][state]['pred']

    source2.data = dict(pc1=X_test[:, 0], pc2=X_test[:, 1],
                        color=np.vectorize({0.0: '#036D9B', 1.0: '#E62A05'}.get)(y_test),
                        poisonous=np.vectorize({0.0: 'no', 1.0: 'yes'}.get)(y_test),
                        pred=point_preds)

    background = Image.fromarray(np.uint8(RdYlBu_r(back_preds) * 255))
    background.putalpha(30)
    background = background.transpose(Image.FLIP_TOP_BOTTOM)
    # background = background.resize((164, 177), Image.ANTIALIAS)
    white = Image.new('RGBA', background.size, (255, 255, 255))
    alpha_composite = Image.alpha_composite(white, background)
    alpha_composite.convert('RGB')
    xdim, ydim = alpha_composite.size
    img = np.empty((ydim, xdim), dtype=np.uint32)
    view = img.view(dtype=np.uint8).reshape((ydim, xdim, 4))
    view[:, :, :] = np.flipud(np.asarray(alpha_composite))

    p2.image_rgba(image=[img], x=-3, y=-3, dw=7, dh=7)
    p2.title.text = model + ' ' * 5 + 'Test Accuracy: ' + str(np.round(model_dict[model][state]['score'], 3))


def update_predictions2():
    global X_train, X_test, y_train, y_test, state

    model = model_select.value
    prob = 'Z_prob' if prediction_select.value == 'Probability' else 'Z'

    state = next(states)

    X_train = model_dict[model][state]['X_train']
    X_test = model_dict[model][state]['X_test']
    y_train = model_dict[model][state]['y_train']
    y_test = model_dict[model][state]['y_test']

    back_preds = model_dict[model][state][prob]
    point_preds = model_dict[model][state]['pred']

    source2.data = dict(pc1=X_test[:, 0], pc2=X_test[:, 1],
                        color=np.vectorize({0.0: '#036D9B', 1.0: '#E62A05'}.get)(y_test),
                        poisonous=np.vectorize({0.0: 'no', 1.0: 'yes'}.get)(y_test),
                        pred=point_preds)

    background = Image.fromarray(np.uint8(RdYlBu_r(back_preds) * 255))
    background.putalpha(30)
    background = background.transpose(Image.FLIP_TOP_BOTTOM)
    # background = background.resize((164, 177), Image.ANTIALIAS)
    white = Image.new('RGBA', background.size, (255, 255, 255))
    alpha_composite = Image.alpha_composite(white, background)
    alpha_composite.convert('RGB')
    xdim, ydim = alpha_composite.size
    img = np.empty((ydim, xdim), dtype=np.uint32)
    view = img.view(dtype=np.uint8).reshape((ydim, xdim, 4))
    view[:, :, :] = np.flipud(np.asarray(alpha_composite))

    p2.image_rgba(image=[img], x=-3, y=-3, dw=7, dh=7)
    p2.title.text = model + ' ' * 5 + 'Test Accuracy: ' + str(np.round(model_dict[model][state]['score'], 3))


model_select.on_change('value', update_predictions)
test_split_button.on_click(update_predictions2)
prediction_select.on_change('value', update_predictions)


curdoc().add_root(row(inputs, p2))
curdoc().title = "Mushrooms"
