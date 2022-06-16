import tensorflow.keras.applications.xception as xception
from keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
from load_save_util import *

data = 'pojedyncze'
results = 'results_pojedyncze'

#data = 'wiele'
#results = 'results_wiele'

model = keras.models.load_model('model')

df = load_data_to_dataframe(data)
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
prepare_result_directory(results)
save_test_results(test_data, predicts, results, 0)
save_misclassified(test_data, predicts, results, 0)
