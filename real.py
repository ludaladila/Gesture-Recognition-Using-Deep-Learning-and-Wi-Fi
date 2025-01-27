import time
import tensorflow as tf
from load_dataset import load_data_specfic
from keras.models import load_model
from keras.utils.np_utils import to_categorical
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

class InferenceServer():

    def __init__(self, root):

        self.root = root
        model_path = 'build/mini_test_model_fullfile_3.h5'
        self.model = tf.keras.models.load_model(model_path)
        self.model.summary()
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def inference(self,  file_path, verbose=True):
        gestures = [ 'circle','down', 'left', 'push','stretch']


        data = load_data_specfic(file_path)

        pred_all = []
        start_time = time.time()
        data_processed=data[np.newaxis, ...]

        output = self.model(data_processed)

        predicted_classes = np.argmax(output, axis=1)
        pred_all.extend(predicted_classes.tolist()) # Add predictions to the list
        predicted_gestures = [gestures[i] for i in predicted_classes]

        end_time = time.time()

        if verbose:
            #print(f'Predict: {pred_all}')
            print(f'Inference time: {end_time - start_time:.2f} (sec)')

        for i, gesture in enumerate(predicted_gestures):
            print(f"POST'")
        if len(pred_all) == 1:
            return pred_all[0]
        else:
            return pred_all


if __name__ == '__main__':
    server = InferenceServer('test/')
    server.inference('C:\\Users\\ludal\\PycharmProjects\\pythonProject\\test\\circle\\circle1.pcap')

