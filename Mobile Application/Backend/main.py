from flask import Flask, request
from flask_restful import Resource, Api
from PIL import Image
import numpy as np
from io import BytesIO
from chexnet import ChexNet
from unet import Unet
from heatmap import HeatmapGenerator
import os
import torchvision.transforms as transforms
from torch.autograd import Variable
from google.cloud import storage
import cv2
from skimage import color, morphology
import matplotlib.pyplot as plt
from collections import OrderedDict


DISEASES = np.array(['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
  'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema',
  'Fibrosis', 'Pleural Thickening', 'Hernia'])

def download_image(bucket, image_name):
    download_blob = bucket.blob('upload/%s' % image_name)
    image = Image.open(BytesIO(download_blob.download_as_string()))
    return image

def upload_image(bucket, image_file, image_name):
    upload_blob = bucket.blob(image_name)
    # byte_arr = BytesIO()
    # image.save(byte_arr, format='PNG')
    # upload_blob.upload_from_string(byte_arr.getvalue())
    upload_blob.upload_from_filename(image_file)

def blend_segmentation(image, mask, gt_mask=None, boundary=False, alpha=1):
    w, h = image.size
    color_mask = np.zeros((h, w, 3)) # PIL Image
    if boundary: mask = morphology.dilation(mask, morphology.disk(3)) - mask
    color_mask[mask==1] = [1, 0, 0] # RGB

    if gt_mask is not None:
        gt_boundary = morphology.dilation(gt_mask, morphology.disk(3)) - gt_mask
        color_mask[gt_boundary==1] = [0, 1, 0] # RGB

    image_hsv = color.rgb2hsv(image)
    color_mask_hsv = color.rgb2hsv(color_mask)

    image_hsv[..., 0] = color_mask_hsv[..., 0]
    image_hsv[..., 1] = color_mask_hsv[..., 1] * alpha

    return color.hsv2rgb(image_hsv)

class CXRApi(Resource):
# class Test():
    # unet = Unet()
    chexnet = ChexNet()
    # heatmap_generator = HeatmapGenerator(chexnet)
    # unet.eval();
    chexnet.eval()
    # client = storage.Client.from_service_account_json('cred.json')
    client = storage.Client()
    bucket = client.bucket('demo-xray')


    def post(self):
    # def post(self, image_name):
        image_name = request.json.get('image_name')
        image = download_image(self.bucket, image_name)
        w, h = image.size
        prob = self.chexnet.predict(image)

        # # run through net
        # (t, l, b, r), mask = self.unet.segment(image)
        # cropped_image = image.crop((l, t, r, b))
        # prob = self.chexnet.predict(cropped_image)

        # # save segmentation result
        # blended = blend_segmentation(image, mask)
        # cv2.rectangle(blended, (l, t), (r, b), (255, 0, 0), 5)
        # plt.imsave('blended.jpg', blended)
        # upload_image(self.bucket, 'blended.jpg', 'seg/%s' % image_name)

        # # save cam result
        # c_w, c_h = cropped_image.size
        # heatmap, _ = self.heatmap_generator.from_prob(prob, c_w, c_h)
        # p_l, p_t = l, t
        # p_r, p_b = w-r, h-b
        # heatmap = np.pad(heatmap, ((p_t, p_b), (p_l, p_r)), mode='linear_ramp', end_values=0)
        # heatmap = ((heatmap - heatmap.min()) * (1 / (heatmap.max() - heatmap.min())) * 255).astype(np.uint8)
        # cam = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET) * 0.4 + np.array(image)
        # cv2.imwrite('cam.jpg', cam)
        # upload_image(self.bucket, 'cam.jpg', 'cam/%s' % image_name)

        # top-10 disease
        idx = np.argsort(-prob)
        top_prob = prob[idx[:10]]
        top_prob = map(lambda x: '{0:5.3f}'.format(x), top_prob)
        top_disease = DISEASES[idx[:10]]
        prediction = OrderedDict(zip(top_disease, top_prob)) # dict in python3.7 ok, 3.5 need OrderedDict

        result = {'result': prediction, 'image_name': image_name}
        return result

if __name__ == '__main__':
    # test = Test()
    # test.post('DEFAULT.JPG')
    port = os.environ.get('PORT', 8080)
    app = Flask(__name__)
    api = Api(app)
    api.add_resource(CXRApi, '/cxr')
    app.run('0.0.0.0', port=port, threaded=False, debug=True)
