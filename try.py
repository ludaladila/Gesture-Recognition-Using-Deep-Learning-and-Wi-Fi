from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from load_dataset import load_data
from sklearn.metrics import confusion_matrix
from model import create_CNN1D, create_CNN2D, create_LSTM_model
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping,LearningRateScheduler


def scheduler(epoch, lr):
    if epoch < 20:
        return 0.001
    elif epoch % 20 == 0:
        return lr * 0.1
    else:
        return lr

dataset_dir = "C:\\Users\\ludal\\PycharmProjects\\pythonProject\\library"
gestures =  ['circle','down','left','push','stretch']
data, labels = load_data(dataset_dir)
labels_one_hot = to_categorical(labels, num_classes=len(gestures))

# SPLIT DATASET
train_data, test_data, train_labels, test_labels = train_test_split(data, labels_one_hot, test_size=0.2, random_state=42)
val_data, test_data, val_labels, test_labels = train_test_split(test_data, test_labels, test_size=0.5, random_state=42)


selected_model = input("Enter 1 for Model 1 or 2 or 3 for Model: ")
if selected_model == '1':
    model = create_CNN1D()
#valid is 2
elif selected_model == '2':
    model = create_CNN2D()
elif selected_model == '3':
    model = create_LSTM_model()
else:
    print("Invalid input! Please enter 1 or 2 or 3.")
    selected_model = input("Enter 1 for Model 1 or 2 for Model: ")

lr_scheduler = LearningRateScheduler(scheduler)

early_stopping = EarlyStopping(monitor='val_loss',
                               patience=10,
                               verbose=1,
                               restore_best_weights=True)


history = model.fit(train_data, train_labels, epochs=64, batch_size=64,
                    validation_split=0.2, callbacks=[lr_scheduler, early_stopping])

'''
# 
history = model.fit(train_data, train_labels, epochs=40, batch_size=64,
                    validation_split=0.2, callbacks=[lr_scheduler])
'''
model.save('build/mini_test_model_bench1.h5')

# Evaluate
loss, accuracy = model.evaluate(test_data, test_labels)
print(f'Model - Test Loss: {loss}, Test Accuracy: {accuracy}')

# matrix
def plot_confusion_matrix(predictions, test_labels, gestures):
    predicted_labels = np.argmax(predictions, axis=1)
    true_labels = np.argmax(test_labels, axis=1)
    cm = confusion_matrix(true_labels, predicted_labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=gestures, yticklabels=gestures)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()


predictions = model.predict(test_data) if selected_model == '1' else model.predict(test_data)
plot_confusion_matrix(predictions, test_labels, gestures)
history = history if selected_model == '1' else history


