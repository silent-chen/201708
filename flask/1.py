# -*- coding: utf-8 -*-
import os
from flask import Flask, request, url_for, send_from_directory,render_template
from werkzeug import secure_filename
import requests

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.getcwd()
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    filepath='../static/120A.png'
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            file_url = url_for('uploaded_file',filename=filename)
            file_send={'file':open(filename,'rb')}
            pre=requests.post('http://127.0.0.1:5000',files=file_send)
            return render_template('mnist.html',answer=pre.text,filepath=file_url)
    return render_template('mnist.html',answer='null',filepath=filepath)



if __name__ == '__main__':
    app.run(port=5001)