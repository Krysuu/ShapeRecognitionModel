import glob
import argparse

import tensorflow.keras.applications.xception as xception
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD
from tensorflow import keras
from save_util import *


def prepare_result_directory(result_directory):
    isExist = os.path.exists(result_directory)
    if not isExist:
        os.makedirs(result_directory)

def load_data_to_dataframe(dataset_dir: str):
    image_paths = []
    image_labels = []
    for filename in glob.glob(dataset_dir + '/*/*.jpg'):
        image_paths.append(filename)
        image_label = filename.split('\\')[1]
        image_labels.append(image_label)

    return pd.DataFrame(list(zip(image_paths, image_labels)), columns=['filename', 'label'])


model = keras.models.load_model('model')

df = load_data_to_dataframe('pojedyncze')
test_idg = ImageDataGenerator(preprocessing_function=xception.preprocess_input)
test_data = test_idg.flow_from_dataframe(
    df,
    x_col='filename',
    y_col='label',
    target_size=(299, 299),
    batch_size=1,
    shuffle=False
)

predicts = model.predict(test_data, verbose=True)
prepare_result_directory("results_pojedyncze")
save_test_results(test_data, predicts, "results_pojedyncze", 0)
save_misclassified(test_data, predicts, "results_pojedyncze", 0)
