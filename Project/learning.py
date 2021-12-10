from CTCModel import CTCModel as CTCModel
from google.colab import drive
from Model import model
from fcts import update_dataframe,decode
import matplotlib.pyplot as plt
from Data_generator import DataGenerator
from google.colab import drive

# mount drive to access dataset
drive.mount('/content/drive')

# path to csv file that contains all details and paths of dataset files
train_csv = '/content/drive/MyDrive/Speech_recognition_dataset/train_data.csv' #path to train.csv
path = '/content/drive/MyDrive/Speech_recognition_dataset/data/' 

# retrieve paths of train audio files
train_dataframe,train_path = update_dataframe(train_csv)
train_path = [path + x.replace('\\', '/') for x in train_path]

# define train and validation dataset with number of samples divisible by the batch size
valid_path=train_path[4160:-12]
train_path=train_path[0:4160]



test_csv = '/content/drive/MyDrive/Speech_recognition_dataset/test_data.csv' #path to test.csv
path = '/content/drive/MyDrive/Speech_recognition_dataset/data/' #path to TIMIT data

# retrieve paths of test audio files
test_dataframe,test_path = update_dataframe(test_csv)
test_path = [path + x.replace('\\', '/') for x in test_path]

# define test dataset with number of samples divisible by the batch size
test_path=test_path[0:-16]


"""
Data Generators

"""

# parametres of data used 
data_params = {'window_len' : 128,
               'hop_len' : 127,
               'nfft' : 256,
               'batch_size' : 32,
               'shuffle' : True
               }
print("Data Generation")

# generate training, validation and test data
training_generator = DataGenerator(train_path, **data_params)
validation_generator = DataGenerator(valid_path, **data_params)
test_generator = DataGenerator(test_path, **data_params)


"""
Model training

"""

#Define the network 
model = model()
model.compile(optimizer=Adam(lr=0.0001))

# implement model
history = model.fit(training_generator,epochs=19,validation_data=validation_generator)


print(history.history.keys())

# plot the evolution of the loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()



