import tensorflow.keras.applications.xception as xception
from keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
from load_save_util import *

data = 'zbiory_danych/prawdziwe_zdjecia/pojedyncze'
results = 'results_pojedyncze'

#data = 'prawdziwe_zdjecia_resized/wiele'
#results = 'results_wiele'

binary_class = ""
is_binary = False

class_mode = 'categorical'
if is_binary:
    class_mode = 'binary'

model = keras.models.load_model('model')
df = load_data_to_dataframe(data, binary_class, is_binary)
test_idg = ImageDataGenerator(preprocessing_function=xception.preprocess_input)
test_data = test_idg.flow_from_dataframe(
    df,
    x_col='filename',
    y_col='label',
    class_mode=class_mode,
    target_size=(299, 299),
    batch_size=1,
    shuffle=False
)

predicts = model.predict(test_data, verbose=True)
prepare_result_directory(results)
save_test_results(test_data, predicts, results, 0, is_binary)
save_misclassified(test_data, predicts, results, 0, is_binary, binary_class)
