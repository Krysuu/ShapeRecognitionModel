import tensorflow.keras.applications.xception as xception
from keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
from load_save_util import *

binary_class = ""
is_binary = False

class_mode = 'categorical'
if is_binary:
    class_mode = 'binary'

main_result_path = "porownanie_prawdziwe/"
data_path_1 = 'zbiory_danych/prawdziwe_zdjecia/pojedyncze'
data_path_2 = 'zbiory_danych/prawdziwe_zdjecia/wiele'

model_path = 'model'
model = keras.models.load_model(model_path)

# pojedyncze

df = load_data_to_dataframe(data_path_1, binary_class, is_binary)
test_idg = ImageDataGenerator(preprocessing_function=xception.preprocess_input)
test_data = test_idg.flow_from_dataframe(
    df,
    x_col='filename',
    y_col='label',
    class_mode=class_mode,
    target_size=(224, 224),
    batch_size=1,
    shuffle=False
)
predicts = model.predict(test_data, verbose=True)
result_path = main_result_path + model_path + "/pojedyncze"
prepare_result_directory(result_path)
save_test_results(test_data, predicts, result_path, 0, is_binary)
save_misclassified(test_data, predicts, result_path, 0, is_binary, binary_class)

# wiele

df = load_data_to_dataframe(data_path_2, binary_class, is_binary)
test_idg = ImageDataGenerator(preprocessing_function=xception.preprocess_input)
test_data = test_idg.flow_from_dataframe(
    df,
    x_col='filename',
    y_col='label',
    class_mode=class_mode,
    target_size=(224, 224),
    batch_size=1,
    shuffle=False
)
predicts = model.predict(test_data, verbose=True)
result_path = main_result_path + model_path + "/wiele"
prepare_result_directory(result_path)
save_test_results(test_data, predicts, result_path, 0, is_binary)
save_misclassified(test_data, predicts, result_path, 0, is_binary, binary_class)
