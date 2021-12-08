import numpy as np
import tensorflow as tf
import string
import librosa
from scipy.io import wavfile


class DataGenerator(tf.keras.utils.Sequence):
    """
    Class DataGenerator : to generate data by batch during the learning  'in case of large dataset'
    Code inspired by : https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
    """


    def __init__(self, list_IDs,shuffle=False, batch_size=32, window_len=128,nfft=256,hop_len=127):
        # parametres for data preprocessing and learning
        self.list_IDs = list_IDs
        self.hop_len = hop_len
        self.batch_size = batch_size
        self.nfft= nfft
        self.window_len = window_len
        self.shuffle = shuffle
        self.on_epoch_end()
        self.char_dict =  { ' ': 0,
                            'a': 1,
                            'b' : 2,
                            'c' : 3,
                            'd' : 4,
                            'e' : 5,
                            'f' : 6,
                            'g' : 7,
                            'h' : 8,
                            'i' : 9,
                            'j' : 10,
                            'k' : 11,
                            'l' : 12,
                            'm' : 13,
                            'n' : 14,
                            'o' : 15,
                            'p' : 16,
                            'q' : 17,
                            'r' : 18,
                            's' : 19,
                            't' : 20,
                            'u' : 21,
                            'v' : 22,
                            'w' : 23,
                            'x' : 24,
                            'y' : 25,
                            'z' : 26
                            }
                          
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def get_data (self,data_IDs):
      '''
      get_data : to retrieve audio files and their associated labels
      data_IDs : paths to the files
      return : audio files and labels
      '''

      son = []
      text = []
      for d in data_IDs:
          fe, audio = wavfile.read(d)
          # normalization of audio 
          moyenne = audio.mean()
          std = audio.std()
          audio = (audio-moyenne) / std
          son.append(audio)
          # path to associated label
          text_path = d[:-8] + '.TXT'  
          txt = open( text_path, 'r')
          ligne = txt.readline()

          # normalization of labels
          ponctuation_char = set(string.punctuation)
          result = ''.join([i for i in ligne if (not i.isdigit() and i not in ponctuation_char )]).lstrip().lower().replace("\n", '')
          text.append(result)
      return son, text


    def get_padded_stft(self,audio, maxlen):
        '''
        get_padded_stft : pad stft of audio files with respect to the longest array
        audio: audio files
        maxlen: length of the longest stft array
        return : padded stft array, length of unpadded array
        '''  
        # calculate the stft of the audio signal
        stft_spectro =np.log(np.absolute(librosa.stft(audio.astype(float),  n_fft= self.nfft,hop_length=self.hop_len, win_length=self.window_len)))
        x_len = stft_spectro.shape[1]
        # pad the stft array
        padded_spectro = tf.keras.preprocessing.sequence.pad_sequences(stft_spectro,maxlen=maxlen, dtype='float', padding='post', truncating='post',value=float(255))
        return padded_spectro, x_len

    def extract_features(self, x_data_init):
        '''
        extract_features : calculate stft of audio signals
        x_data_init : audio files
        return : padded and transposed stft arrays, length of unpadded arrays
        '''  
        stft_spectrogram = np.log(np.absolute(librosa.stft(max(x_data_init, key=len).astype(float),n_fft= self.nfft,hop_length=self.hop_len, win_length=self.window_len)))
        x_max_length = stft_spectrogram.shape[1]

        x_data = []
        x_data_len = []

        for i in range(len(x_data_init)):

            padded_spectro, x_len = self.get_padded_stft(x_data_init[i], maxlen=x_max_length)

            x_data.append(padded_spectro.T)
            x_data_len.append(x_len) 

        data_input = np.array(x_data)
        input_length = np.array(x_data_len)

        return data_input,input_length

    def encode_text(self, y_data_init):
        '''
        encode_text : encode labels in digits
        y_data_init : labels "text"
        return : padded labels, length of labels before padding
        '''  
        y_max_length = len(max(y_data_init, key=len))
        y_data = []
        y_data_len = []

        for i in range(len(y_data_init)):
            unpadded_y = []
            for c in y_data_init[i]:
                if c=='':
                    unpadded_y.append(self.char_dict['<SPACE>'])
                else:
                  unpadded_y.append(self.char_dict[c])
            y_len = len(unpadded_y)
            for j in range(len(unpadded_y), y_max_length):
                    unpadded_y.append(float(255))
            y_data.append(unpadded_y)
            y_data_len.append(y_len)

        # convert to array
        y_data = np.array(y_data)
        label_length = np.array(y_data_len)


        return y_data, label_length


    def __data_generation(self, list_IDs_temp):
        """
        data_generator : to generate data by batch during the learning  'in case of large dataset'
        list_IDs_temp : paths to the dataset
        return batchs of data
        """
        x_data_init, y_data_init = self.get_data(list_IDs_temp)
        x_data, input_length = self.extract_features(x_data_init)
        y_data, label_length = self.encode_text(y_data_init)

        inputs = [x_data, y_data, input_length, label_length]
        outputs = np.zeros([self.batch_size])
        return inputs,outputs