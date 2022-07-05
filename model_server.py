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

model = keras.models.load_model('models/8k_standard')
current_model_name = 'models/8k_standard'


def predictWithModel(img, model_name, is_binary, class_name):
    global model, current_model_name
    if current_model_name != model_name:
        model = keras.models.load_model(model_name)
        current_model_name = model_name

    img = img.resize((299, 299), Image.ANTIALIAS)
    x = np.array(img)
    x = np.expand_dims(x, axis=0)
    x = xception.preprocess_input(x)

    predicted = model.predict(x)

    if is_binary:
        processed = [{'label': class_name, 'score': float(predicted[0][0])}]
        return json.dumps(processed)
    else:
        preds = decode_predictions(predicted, top=8)[0]
        processed = [({'label': label, 'score': float(score)}) for __, label, score in preds]
        return json.dumps(processed)


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


def get_image_from_request(req):
    input_file = req.get_param('file')
    raw = input_file.file.read()
    return Image.open(io.BytesIO(raw))


class StandardResource(object):
    def on_post(self, req, resp):
        image = get_image_from_request(req)
        resp.body = predictWithModel(image, 'models/8k_standard', False, None)


class CeownikResource(object):
    def on_post(self, req, resp):
        image = get_image_from_request(req)
        resp.body = predictWithModel(image, 'models/ceownik', True, 'ceownik')

class DwuteownikResource(object):
    def on_post(self, req, resp):
        image = get_image_from_request(req)
        resp.body = predictWithModel(image, 'models/dwuteownik', True, 'dwuteownik')

class KatownikResource(object):
    def on_post(self, req, resp):
        image = get_image_from_request(req)
        resp.body = predictWithModel(image, 'models/katownik', True, 'katownik')

class KwadratowyResource(object):
    def on_post(self, req, resp):
        image = get_image_from_request(req)
        resp.body = predictWithModel(image, 'models/kwadratowy', True, 'kwadratowy')

class OkraglyResource(object):
    def on_post(self, req, resp):
        image = get_image_from_request(req)
        resp.body = predictWithModel(image, 'models/okragly', True, 'okragly')

class PlaskownikResource(object):
    def on_post(self, req, resp):
        image = get_image_from_request(req)
        resp.body = predictWithModel(image, 'models/plaskownik', True, 'plaskownik')

class ProfilResource(object):
    def on_post(self, req, resp):
        image = get_image_from_request(req)
        resp.body = predictWithModel(image, 'models/profil', True, 'profil')

class RuraResource(object):
    def on_post(self, req, resp):
        image = get_image_from_request(req)
        resp.body = predictWithModel(image, 'models/rura', True, 'rura')


app = application = falcon.API(middleware=[MultipartMiddleware()])
app.add_route('/models/standard', StandardResource())
app.add_route('/models/ceownik', CeownikResource())
app.add_route('/models/dwuteownik', DwuteownikResource())
app.add_route('/models/katownik', KatownikResource())
app.add_route('/models/kwadratowy', KwadratowyResource())
app.add_route('/models/okragly', OkraglyResource())
app.add_route('/models/plaskownik', PlaskownikResource())
app.add_route('/models/profil', ProfilResource())
app.add_route('/models/rura', RuraResource())

if __name__ == '__main__':
    with make_server('', 8000, app) as httpd:
        print('Serving on port 8000...')
        # Serve until process is killed
        httpd.serve_forever()
