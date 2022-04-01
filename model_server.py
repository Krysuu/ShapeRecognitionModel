#!/usr/bin/python3

import json
import io

from keras.preprocessing import image
from keras.applications import inception_v3 as incep
from falcon_multipart.middleware import MultipartMiddleware
from wsgiref.simple_server import make_server
from PIL import Image
import numpy as np
import falcon


model = incep.InceptionV3(weights='imagenet', classes=1000)

def predict(img: Image):
    img = img.resize((299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = incep.preprocess_input(x)

    preds = incep.decode_predictions(model.predict(x), top=5)[0]
    return preds

def prepare_predictions(preds):
    processed = [({'label': label, 'score': float(score)}) for __, label, score in preds]
    return json.dumps(processed)

class ImageResource(object):
    def on_post(self, req, resp):
        input_file = req.get_param('file')
        raw = input_file.file.read()

        image = Image.open(io.BytesIO(raw))
        resp.body = prepare_predictions(predict(image))

app = application = falcon.API(middleware=[MultipartMiddleware()])
app.add_route('/models/image', ImageResource())

if __name__ == '__main__':
    with make_server('', 8000, app) as httpd:
        print('Serving on port 8000...')

        # Serve until process is killed
        httpd.serve_forever()