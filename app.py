from flask import Flask, render_template, send_file
from bokeh.embed import autoload_server
from bokeh.client import pull_session
from werkzeug.contrib.fixers import ProxyFix


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
    url = 'http://104.131.82.22:5006'
    session = pull_session(url=url, app_path='/mushroom_classifier')
    bokeh_script = autoload_server(None, app_path='/mushroom_classifier', session_id=session.id, url=url)
    return render_template("mushroomclassifier.html", bokeh_script=bokeh_script)


app.wsgi_app = ProxyFix(app.wsgi_app)

if __name__ == '__main__':

    app.run(debug=True)
