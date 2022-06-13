#!/usr/bin/python3

import json
import io

import tensorflow.keras.applications.xception as xception
from falcon_multipart.middleware import MultipartMiddleware
from wsgiref.simple_server import make_server
from PIL import Image
import numpy as np
import falcon
from tensorflow import keras

model = keras.models.load_model('model')


def predict(img: Image):
    img = img.resize((299, 299))
    x = np.array(img)
    x = np.expand_dims(x, axis=0)
    x = xception.preprocess_input(x)

    predicted = model.predict(x)
    preds = decode_predictions(predicted, top=8)[0]
    return preds


def decode_predictions(preds, top=8, class_list_path='klasy.json'):
    if len(preds.shape) != 2 or preds.shape[1] != 8:  # your classes number
        raise ValueError('`decode_predictions` expects '
                         'a batch of predictions '
                         '(i.e. a 2D array of shape (samples, 1000)). '
                         'Found array with shape: ' + str(preds.shape))
    index_list = json.load(open(class_list_path))
    results = []
    for pred in preds:
        top_indices = pred.argsort()[-top:][::-1]
        result = [tuple(index_list[str(i)]) + (pred[i],) for i in top_indices]
        result.sort(key=lambda x: x[2], reverse=True)
        results.append(result)
    return results


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
