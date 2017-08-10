from flask import Flask, render_template, url_for, send_file
from bokeh.embed import autoload_server
from bokeh.client import pull_session
import subprocess


app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/resume')
def plot_csv():
    return send_file('/Users/filip/PycharmProjects/filip_site/static/Resume/Filip_Jankovic_Resume.pdf')

@app.route('/pandasseaborn')
def pandasseaborn():
    return render_template('pandasseaborn.html')

@app.route('/classifiervisualization')
def classifiervisualization():
    return render_template('classifiervisualization.html')

@app.route('/pandasregex')
def pandasregex():
    return render_template('pandasregex.html')

@app.route('/sentiment')
def sentiment():
    return render_template('sentimentanalysis.html')

@app.route('/networkviz')
def networkviz():
    return render_template('networkvisualization.html')

@app.route('/interactive')
def interactive():
    bokeh_script = autoload_server(url='http://localhost:5006/mushroom_classifier')
    return render_template('mushroomclassifier.html', bokeh_script=bokeh_script)


if __name__ == '__main__':

    cmd = "bokeh serve --allow-websocket-origin=localhost:5000 --allow-websocket-origin=localhost:5006 " \
          "mushroom_classifier.py "
    bokeh_serve = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
    app.run()
    bokeh_serve.kill()
