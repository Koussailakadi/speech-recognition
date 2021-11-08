# les etapes :
1. organiser les données :
 1.1. dézipper 
 1.2. organiser les dossiers
2. model data input :
 2.1. calculer spectrogram de chaque son
 2.2. feature extraction avec librosa (fct melspectrogram)
3. architecture model:
  LSTM bidirectionnel + CTC 
4. decodage de résultat:
  Beam search algorithm
5.  test
