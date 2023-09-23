from flask import Flask, render_template, request
import subprocess

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    output = ""
    if request.method == 'POST':
        code = request.form['code']
        process = subprocess.Popen(
            ['./jaithon'],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        output, _ = process.communicate(input=code)
    return render_template('index.html', output=output)

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=8080)
