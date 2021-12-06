from CTCModel import CTCModel as CTCModel
from google.colab import drive
from Model import model
from fcts import update_dataframe,decode
import matplotlib.pyplot as plt
from Data_generator import DataGenerator
from google.colab import drive


drive.mount('/content/drive')
# !cp -r /content/drive/MyDrive/Speech_recognition_dataset  /content/
# drive.mount('/content/drive')
train_csv = '/content/drive/MyDrive/Speech_recognition_dataset/train_data.csv' #path to train.csv
path = '/content/drive/MyDrive/Speech_recognition_dataset/data/' #path to TIMIT data
train_dataframe,train_path = update_dataframe(train_csv)
train_path = [path + x.replace('\\', '/') for x in train_path]
valid_path=train_path[4160:-12]
train_path=train_path[0:4160]



test_csv = '/content/drive/MyDrive/Speech_recognition_dataset/test_data.csv' #path to test.csv
path = '/content/drive/MyDrive/Speech_recognition_dataset/data/' #path to TIMIT data
test_dataframe,test_path = update_dataframe(test_csv)
test_path = [path + x.replace('\\', '/') for x in test_path]
test_path=test_path[0:-16]

"""
Data Generators

"""


data_params = {'window_len' : 128,
               'hop_len' : 127,
               'nfft' : 256,
               'batch_size' : 32,
               'shuffle' : True
               }
print("Data Generation")
training_generator = DataGenerator(train_path, **data_params)
validation_generator = DataGenerator(valid_path, **data_params)
test_generator = DataGenerator(test_path, **data_params)


"""
Model training

"""
model = model()
model.compile(optimizer=Adam(lr=0.0001))
# print(type(model))

history = model.fit(training_generator,epochs=5,validation_data=validation_generator)



print(history.history.keys())

# summarize
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()



"""
Model evaluation

"""

x_data_init, y_data_init = test_generator.get_data(test_path[0:32])
x_data, input_length = test_generator.extract_features(x_data_init)
y_data, label_length = test_generator.encode_text(y_data_init)

print("Evaluation")

model.evaluate([x_data, y_data, input_length, label_length],batch_size=32,metrics=['ler', 'ser'])


print("Prediction")
pred = model.predict_generator(test_generator,steps=1)


y_pred=decode(pred[0][3])
print(y_pred)

y_true=decode(pred[1][3])
print(y_true)
