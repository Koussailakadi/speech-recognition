from CTCModel import CTCModel as CTCModel
import pandas as pd




def update_dataframe(csv_path):
    '''
    update_dataframe : function to fetch audio files titles from the dataset and their paths
    csv_path : the path to the csv file that contains all the details and paths of the files of the dataset

    return : audio files names and their paths
    '''
    ## garder que les fichiers audios

    data = pd.read_csv(csv_path)
    data = data.dropna(subset=['filename'])
    data = data.drop(['test_or_train', 'dialect_region', 'filename',
                      'path_from_data_dir', 'is_audio', 'is_word_file',
                      'is_phonetic_file', 'is_sentence_file',
                      'speaker_id', 'index'], axis=1)

    data = data[(data.is_converted_audio == True)]
    data = data.reset_index(drop=True)
    paths = list(data['path_from_data_dir_windows'])
    return data,paths


def decode(sequence):

    '''
    decode : function to decode the phrase coded in digits
    label_pred : the coded predicted phrase
    return : phrase
    '''
    unpaded = [j for j in sequence if j != -1]
    pred = []
    char_dict =  { ' ': 0,
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
    char_dict_inv= dict((v,k) for k,v in char_dict.items())
    for c in unpaded:
        if c == 0:
            pred.append(" ")
        if c==255:
          pred.append("")
        else:
            pred.append(char_dict_inv[c])
    pred = ''.join(pred)
    return pred