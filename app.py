from flask import Flask, render_template, url_for, send_file

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


if __name__ == '__main__':
    app.run(debug=True)
