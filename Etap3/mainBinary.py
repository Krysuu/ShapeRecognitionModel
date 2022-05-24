import glob
import argparse
from timeit import default_timer as timer

import tensorflow.keras.applications.xception as xception
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.optimizers import SGD
import math

from save_util import *

class TimingCallback(Callback):
    def __init__(self, logs={}):
        super().__init__()
        self.logs = []

    def on_epoch_begin(self, epoch, logs={}):
        self.starttime = timer()

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(timer() - self.starttime)

    def clear(self):
        self.logs = []


time_cb = TimingCallback()


def prepare_result_directory(result_directory):
    isExist = os.path.exists(result_directory)
    if not isExist:
        os.makedirs(result_directory)


def load_data_to_dataframe(dataset_dir: str):
    image_paths = []
    image_labels = []
    for filename in glob.glob(dataset_dir + '/*/*.png'):
        image_paths.append(filename)
        image_label = filename.split('\\')[1]
        image_labels.append(image_label)

    return pd.DataFrame(list(zip(image_paths, image_labels)), columns=['filename', 'label'])


def preprocess_and_load_data(train_dataframe, test_dataframe, preprocessing_function, batch_size):
    train_idg = ImageDataGenerator(preprocessing_function=preprocessing_function)
    test_idg = ImageDataGenerator(preprocessing_function=preprocessing_function)

    train_data = train_idg.flow_from_dataframe(
        train_dataframe,
        x_col='filename',
        y_col='label',
        class_mode='binary',
        target_size=(ROWS, COLS),
        batch_size=batch_size
    )

    test_data = test_idg.flow_from_dataframe(
        test_dataframe,
        x_col='filename',
        y_col='label',
        class_mode='binary',
        target_size=(ROWS, COLS),
        batch_size=1,
        shuffle=False
    )
    return train_data, test_data


def prepare_model(pretrained_model, learning_rate, momentum, dropout, dense_size):
    model = Sequential()
    model.add(pretrained_model)
    model.add(Flatten())
    model.add(Dense(dense_size, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer=SGD(learning_rate=learning_rate, momentum=momentum),
                  metrics=['accuracy'])
    model.summary()
    return model


def fit_model(model, train_gen, weights_path, epochs):
    checkpoint = ModelCheckpoint(weights_path, monitor='loss', verbose=1, save_best_only=True, mode='min')
    early = EarlyStopping(monitor='loss', patience=3)

    callbacks_list = [checkpoint, early, time_cb]
    return model.fit(train_gen,
                     epochs=epochs,
                     shuffle=True,
                     verbose=2,
                     callbacks=[callbacks_list]
                     )


def perform_test(preprocess_input, pretrained_model, n_splits, data_directory, result_directory, dense_size, batch_size,
                 learning_rate, momentum):
    df = load_data_to_dataframe(data_directory)
    nclass = df['label'].nunique()
    ceownik_df = df.loc[df['label'] == 'ceownik']
    n_images = len(ceownik_df.index)
    other_df = df.loc[df['label'] != 'ceownik']
    other_df = other_df.groupby('label').head(math.floor(n_images/(nclass-1)))
    other_df = other_df.assign(label='inny')
    df = pd.concat([ceownik_df, other_df])

    prepare_result_directory(result_directory)

    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1)
    current_fold = 0
    for fold in kf.split(df, df['label']):
        current_fold += 1
        if skip_folds_after != 0 and current_fold > skip_folds_after:
            break

        print(f'Current fold: {current_fold}/{n_splits}')
        train_df = df.iloc[fold[0]]
        test_df = df.iloc[fold[1]]

        train_gen, test_gen = preprocess_and_load_data(train_dataframe=train_df,
                                                       test_dataframe=test_df,
                                                       preprocessing_function=preprocess_input,
                                                       batch_size=batch_size
                                                       )

        model = prepare_model(pretrained_model,
                              learning_rate,
                              momentum,
                              dropout,
                              dense_size
                              )

        history = fit_model(model=model,
                            train_gen=train_gen,
                            weights_path=weights_path,
                            epochs=epochs,
                            )

        save_training_progress(history=history, current_fold=current_fold, result_directory=result_directory)

        model.load_weights(weights_path)
        predicts = model.predict(test_gen, verbose=True)

        save_test_results_binary(test_gen=test_gen, predicts=predicts, result_directory=result_directory,
                          current_test=current_fold)
        save_misclassified_binary(test_gen=test_gen,
                                  predicts=predicts,
                                  result_directory=result_directory,
                                  current_fold=current_fold
                                  )

        save_and_reset_time_logs(result_directory, time_cb)

# Constant
ROWS = 299
COLS = 299
n_splits = 5
weights_path = "weights.best.hdf5"
epochs = 50
dropout = 0.5
skip_folds_after = 0

parser = argparse.ArgumentParser()
parser.add_argument('dataset_path', type=str, help='path to dataset directory')
parser.add_argument('result_path', type=str, help='path to result directory')
parser.add_argument('--dense_size', type=int, default=256, help='Size of final dense layer')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning_rate')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
# parser.add_argument('--decay_rate', type=float, default=0.00001, help='Decay rate')
args = vars(parser.parse_args())

pretrained_model = xception.Xception(weights="imagenet",
                                     include_top=False,
                                     input_shape=(ROWS, COLS, 3)
                                     )
pretrained_model.trainable = False

perform_test(xception.preprocess_input,
             pretrained_model,
             n_splits,
             args['dataset_path'],
             args['result_path'],
             args['dense_size'],
             args['batch_size'],
             args['learning_rate'],
             args['momentum']
             )
