from keras.models import load_model
from keras.utils.np_utils import to_categorical
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from load_dataset import load_data_test


dataset_dir = "C:\\Users\\ludal\\PycharmProjects\\pythonProject\\data_mar18"
new_data, new_labels = load_data_test(dataset_dir)
gestures = ['circle','down','left','push','stretch']


new_labels_one_hot = to_categorical(new_labels, num_classes=len(gestures))

# trained model
model = load_model('build/mini_test_model_3.h5')
new_predictions = model.predict(new_data)

# evaluate
new_loss, new_accuracy = model.evaluate(new_data, new_labels_one_hot)
print(f'New Dataset - Loss: {new_loss}, Accuracy: {new_accuracy}')


def plot_confusion_matrix(predictions, true_labels, gestures):
    predicted_labels = np.argmax(predictions, axis=1)
    true_labels = np.argmax(true_labels, axis=1)
    cm = confusion_matrix(true_labels, predicted_labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=gestures, yticklabels=gestures)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()

plot_confusion_matrix(new_predictions, new_labels_one_hot, gestures)
