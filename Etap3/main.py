import argparse

import tensorflow.keras.applications.xception as xception
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.optimizers import SGD

from load_save_util import *

time_cb = TimingCallback()


def scheduler(epoch, lr):
    if epoch < 1:
        return lr
    else:
        return lr * decay_rate


def preprocess_and_load_data(train_dataframe, test_dataframe, preprocessing_function, batch_size, is_binary):
    train_idg = ImageDataGenerator(preprocessing_function=preprocessing_function)
    test_idg = ImageDataGenerator(preprocessing_function=preprocessing_function)

    class_mode = 'categorical'
    if is_binary:
        class_mode = 'binary'

    train_data = train_idg.flow_from_dataframe(
        train_dataframe,
        x_col='filename',
        y_col='label',
        class_mode=class_mode,
        target_size=(ROWS, COLS),
        batch_size=batch_size
    )

    test_data = test_idg.flow_from_dataframe(
        test_dataframe,
        x_col='filename',
        y_col='label',
        class_mode=class_mode,
        target_size=(ROWS, COLS),
        batch_size=1,
        shuffle=False
    )
    return train_data, test_data


def prepare_model(pretrained_model, train_gen, learning_rate, momentum, dropout, dense_size, is_binary):
    model = Sequential()
    model.add(pretrained_model)
    model.add(Flatten())
    model.add(Dense(dense_size, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(dense_size, activation='relu'))
    model.add(Dropout(dropout))

    if is_binary:
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy',
                      optimizer=SGD(learning_rate=learning_rate, momentum=momentum),
                      metrics=['accuracy'])
    else:
        model.add(Dense(len(train_gen.class_indices), activation='softmax'))
        model.compile(loss='categorical_crossentropy',
                      optimizer=SGD(learning_rate=learning_rate, momentum=momentum),
                      metrics=['accuracy'])

    model.summary()
    return model


def fit_model(model, train_gen, weights_path, epochs):
    checkpoint = ModelCheckpoint(weights_path, monitor='loss', verbose=1, save_best_only=True, mode='min')
    early = EarlyStopping(monitor='loss', patience=3)
    lr_callback = LearningRateScheduler(scheduler, verbose=0)

    callbacks_list = [checkpoint, early, time_cb, lr_callback]
    return model.fit(train_gen,
                     epochs=epochs,
                     shuffle=True,
                     verbose=2,
                     callbacks=[callbacks_list],
                     workers=4
                     )


def perform_test(preprocess_input, pretrained_model, n_splits, data_directory, result_directory, dense_size, batch_size,
                 learning_rate, momentum, binary_class, is_binary):
    df = load_data_to_dataframe(data_directory, binary_class, is_binary)
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
                                                       batch_size=batch_size,
                                                       is_binary=is_binary
                                                       )

        model = prepare_model(pretrained_model,
                              train_gen,
                              learning_rate,
                              momentum,
                              dropout,
                              dense_size,
                              is_binary
                              )

        history = fit_model(model=model,
                            train_gen=train_gen,
                            weights_path=weights_path,
                            epochs=epochs,
                            )

        save_training_progress(history=history, current_fold=current_fold, result_directory=result_directory)

        model.load_weights(weights_path)
        predicts = model.predict(test_gen, verbose=True)

        save_test_results(test_gen=test_gen, predicts=predicts, result_directory=result_directory,
                          current_test=current_fold, is_binary=is_binary)
        save_misclassified(test_gen=test_gen,
                           predicts=predicts,
                           result_directory=result_directory,
                           current_fold=current_fold,
                           is_binary=is_binary,
                           binary_class=binary_class
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
parser.add_argument('--dense_size', type=int, default=128, help='Size of final dense layer')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning_rate')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
parser.add_argument('--decay_rate', type=float, default=0.94, help='Decay rate')
parser.add_argument('--binary_class', type=str, default="", help='Binary class')
args = vars(parser.parse_args())

decay_rate = args['decay_rate']
pretrained_model = xception.Xception(weights="imagenet",
                                     include_top=False,
                                     input_shape=(ROWS, COLS, 3)
                                     )
pretrained_model.trainable = False

binary_class = args['binary_class']
is_binary = False
if binary_class != "":
    is_binary = True

perform_test(xception.preprocess_input,
             pretrained_model,
             n_splits,
             args['dataset_path'],
             args['result_path'],
             args['dense_size'],
             args['batch_size'],
             args['learning_rate'],
             args['momentum'],
             binary_class,
             is_binary,
             )
