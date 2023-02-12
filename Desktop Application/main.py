from PIL import Image
import numpy as np
from chexnet import ChexNet
import os
from flask import Flask, render_template, request
from werkzeug import secure_filename
import csv



DISEASES = np.array(['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
  'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema',
  'Fibrosis', 'Pleural Thickening', 'Hernia'])

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'C:/Users/bhave/OneDrive/Desktop/Sakshi/images/'
chexnet = ChexNet()
chexnet.eval()

def pred(fname):
   image_path = 'C:/Users/bhave/OneDrive/Desktop/Sakshi/images/'+fname
   image = Image.open(image_path)
   image = image.convert('RGB')
   prob = chexnet.predict(image)
   idx = np.argsort(-prob)
   top_prob = prob[idx[:10]]
   top_prob = top_prob*100
   top_prob = np.round(top_prob,2)
   top_disease = DISEASES[idx[:10]]
   result = {}
   with open('Book1.csv', 'rt') as f:
    reader = csv.reader(f)
    for r in reader:
       ele = r[0]
       if(ele == fname):
          if(r[3]!=''):
             actual = r[1]+' or '+r[2]+' or '+r[3]
          if(r[2]!=''):
             actual = r[1]+' or '+r[2]
          else:
             actual = r[1]   
          print(actual)


      
 
   for A,B in zip(top_disease,top_prob):
      result[A] = B


   return render_template('result.html',result = result,actual = actual)


@app.route('/')
def upload_f():
   return render_template('front.html')

@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
      f = request.files['file']
      f.save(os.path.join(app.config['UPLOAD_FOLDER'],secure_filename(f.filename)))
      return pred(f.filename)

##print(result)

if __name__ == '__main__':
    app.run(debug=True)
