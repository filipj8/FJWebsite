from flask import Flask, render_template, send_file
from bokeh.embed import autoload_server
from bokeh.client import pull_session


app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/resume')
def resume():
    return send_file('./static/Resume/FilipJankovicResume.pdf', attachment_filename='FilipJankovicResume.pdf')


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
    session = pull_session(url="http://localhost:5006/mushroom_classifier")
    bokeh_script = autoload_server(None, url="http://localhost:5006/mushroom_classifier", session_id=session.id)
    return render_template("mushroomclassifier.html", bokeh_script=bokeh_script)


if __name__ == '__main__':

    app.run(debug=True)
