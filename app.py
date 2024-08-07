from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image  # Pastikan Pillow diimpor

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'

model = load_model('model_vgg16_transfer_learning.h5')

class_names = ['Bandhil', 'Candrasa', 'Keris', 'Patrem', 'Tombak', 'Tulup', 'Wedhung']

data = {
    'Bandhil': {
        'image': 'images/bandhil.jpg',
        'article': 'Bandhil adalah senjata yang digunakan dengan cara dipukul atau dilemparkan. Senjata ini terdiri dari peluru yang biasanya terbuat dari besi dan rantai panjang yang terbuat dari anyaman tampar. Bandhil sering digunakan untuk berburu dan juga sebagai alat pertahanan diri. Pembuatan bandhil melibatkan teknik anyaman dan penempaan .'
    },
    'Candrasa': {
        'image': 'images/candrasa.jpeg',
        'article': 'Candrasa adalah senjata yang bentuknya mirip dengan hiasan sanggul kepala atau tusuk konde. Candrasa sering digunakan oleh prajurit wanita sebagai senjata rahasia. Bentuknya yang kecil dan mirip tusuk konde memudahkan prajurit wanita untuk menyembunyikannya di sanggul mereka. Candrasa terbuat dari besi dan sangat runcing serta bercabang-cabang .'
    },
        'Keris': {
        'image': 'images/keris.png',
        'article': 'Keris adalah salah satu senjata tradisional yang paling terkenal di Indonesia, khususnya di Yogyakarta. Senjata ini tidak hanya berfungsi sebagai alat pertahanan tetapi juga memiliki nilai budaya dan spiritual. Keris dibuat oleh seorang empu dengan teknik tempa dan bakar, menggunakan berbagai jenis logam. Keris biasanya memiliki bilah yang berlekuk-lekuk dan dihiasi dengan pamor, yang dipercaya memiliki kekuatan magis .'
    },
        'Patrem': {
        'image': 'images/Patrem.jpeg',
        'article': 'Patrem adalah senjata kecil yang mirip dengan keris tetapi berukuran lebih kecil. Patrem sering digunakan oleh wanita sebagai alat pertahanan diri dan juga sebagai hiasan. Senjata ini juga dibuat oleh empu dengan teknik yang sama seperti pembuatan keris, dan sering dihiasi dengan pamor serta ornamen lainnya .'
    },
        'Tombak': {
        'image': 'images/tombak.jpg',
        'article': 'Tombak adalah senjata yang dikenal baik oleh masyarakat Jawa setelah keris. Tombak memiliki bilah besi yang meruncing di ujungnya dan digunakan baik untuk berperang maupun berburu. Tombak juga dibuat oleh empu dengan teknik tempa dan bakar, dan sering kali dihiasi dengan permata pada selutnya. Ada berbagai jenis tombak, termasuk tombak yang dianggap bertuah dan memiliki kekuatan gaib.'
    },
        'Tulup': {
        'image': 'images/tulup.jpeg',
        'article': 'Tulup atau sumpit adalah senjata tradisional yang digunakan dengan cara meniupkan peluru kecil dari bambu melalui sebuah tabung panjang. Tulup sering digunakan untuk berburu burung. Bahan baku utama tulup adalah bambu kecil yang dilubangi dan dibersihkan untuk membuat jalan bagi peluru yang terbuat dari tanah atau benda kecil lainnya .'
    },
        'Wedhung': {
        'image': 'images/wedhung.jpg',
        'article': 'Wedhung adalah senjata yang mirip dengan pisau dapur besar dan sering digunakan oleh para abdi dalem Kraton Yogyakarta. Senjata ini terbuat dari besi dan baja, dan ada juga yang dihiasi dengan pamor. Wedhung memiliki sarung yang terbuat dari kayu, biasanya trembalo atau cendhono, dan dihiasi dengan berbagai ornamen seperti emas atau perak.'
    },
}

def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0
    preds = model.predict(x)
    predicted_class = class_names[np.argmax(preds, axis=1)[0]]
    return predicted_class

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            predicted_class = model_predict(filepath, model)
            return render_template('result.html', result=predicted_class, data=data, image_src=file.filename)
    return render_template('upload.html')

@app.route('/artikel')
def artikel():
    return render_template('artikel.html')

@app.route('/tentang')
def tentang():
    return render_template('tentang.html')

if __name__ == '__main__':
    app.run(debug=True)
