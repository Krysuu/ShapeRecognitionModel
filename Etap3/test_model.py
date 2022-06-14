import glob
import argparse

import tensorflow.keras.applications.xception as xception
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD

from save_util import *


def load_data_to_dataframe(dataset_dir: str):
    image_paths = []
    image_labels = []
    for filename in glob.glob(dataset_dir + '/*/*.png'):
        image_paths.append(filename)
        image_label = filename.split('\\')[1]
        image_labels.append(image_label)

    return pd.DataFrame(list(zip(image_paths, image_labels)), columns=['filename', 'label'])


def start(data_directory, dense_size, batch_size, learning_rate, momentum):
    df = load_data_to_dataframe(data_directory)
    train_idg = ImageDataGenerator(preprocessing_function=xception.preprocess_input)

    train_data = train_idg.flow_from_dataframe(
        df,
        x_col='filename',
        y_col='label',
        target_size=(ROWS, COLS),
        batch_size=batch_size
    )

    pretrained_model = xception.Xception(weights="imagenet",
                                         include_top=False,
                                         input_shape=(ROWS, COLS, 3)
                                         )
    pretrained_model.trainable = False

    nclass = 8
    model = Sequential()
    model.add(pretrained_model)
    model.add(Flatten())
    model.add(Dense(dense_size, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(nclass, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=SGD(learning_rate=learning_rate, momentum=momentum),
                  metrics=['accuracy'])
    model.summary()

    checkpoint = ModelCheckpoint(weights_path, monitor='loss', verbose=1, save_best_only=True, mode='min')
    early = EarlyStopping(monitor='loss', patience=3)

    callbacks_list = [checkpoint, early]
    model.fit(train_data,
              epochs=epochs,
              shuffle=True,
              verbose=2,
              callbacks=[callbacks_list]
              )

    model.save('model')


# Constant
ROWS = 299
COLS = 299
n_splits = 5
weights_path = "weights.best.hdf5"
epochs = 9999
dropout = 0.5
skip_folds_after = 0

parser = argparse.ArgumentParser()
parser.add_argument('dataset_path', type=str, help='path to dataset directory')
parser.add_argument('--dense_size', type=int, default=256, help='Size of final dense layer')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning_rate')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
args = vars(parser.parse_args())

start(args['dataset_path'],
      args['dense_size'],
      args['batch_size'],
      args['learning_rate'],
      args['momentum'])
