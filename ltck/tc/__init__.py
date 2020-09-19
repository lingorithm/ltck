import llck
import numpy as np
import os
from sklearn.preprocessing import OneHotEncoder
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Bidirectional, Embedding, Dropout
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split

class TC:
    """Text Classification Class
    """
    def __init__(self, lang, data, **kwargs):
        """init TC

        Args:
            lang (str): language for the classifier used for llck
            data (list): list of tuples 
        """
        self.nlp = llck(lang, {'tokenize': 'tokenize'})
        self.data = data
        self.lang = lang
        self.verbose = kwargs.get('verbose', False)
        self.sentences = []
        self.classes = []
        self._uniqe_classes = set([])
        self.__pre_processing()
        if self.verbose:
            self.data_summary()

    def __pre_processing(self):
        """Pre pcrocess data before training 
        """
        for tup in self.data:
            self.sentences.append(list(self.nlp.process(tup[0]).sentences[0].tokens))
            self.classes.append(tup[1])
            self._uniqe_classes.add(tup[1])
        # get max length for sentence
        self._max_len = len(max(self.sentences, key=len))
        # pre process x 
        self.__pre_processing_x()
        # pre process y
        self.__pre_processing_y()
        # get vocab size
        self._vocab_size = len(self.x_token.word_index) + 1

    def __pre_processing_x(self):
        """Sequence data and padding sentence for X
        """
        self.x_token = self.__tokenizer(self.sentences)
        self.x_encoded = self.x_token.texts_to_sequences(self.sentences)
        self.x = pad_sequences(self.x_encoded, maxlen=self._max_len,
                      padding="post")

    def __pre_processing_y(self):
        """Sequence data and hot encode classes for Y
        """
        y_token = self.__tokenizer(self._uniqe_classes)
        y_encoded = y_token.texts_to_sequences(self.classes)
        y_encoded = np.array(
            y_encoded).reshape(len(y_encoded), 1)
        o = OneHotEncoder(sparse=False)
        self.y = o.fit_transform(y_encoded)

    def __tokenizer(self, input):
        """Tokenize the data to be used as sequence

        Args:
            input (list): list of data

        Returns:
            Token Object
        """
        token = Tokenizer(filters='')
        token.fit_on_texts(input)
        return token

    def data_summary(self):
        """Print data summary
        """
        print("Data Summary")
        print("Vocab Size = %d and Maximum length = %d" %
              (self._vocab_size, self._max_len))
        print("Sentence Size = %d and Classes Size = %d" %
              (len(self.sentences), len(self.uniqe_classes)))
        print("Classes = ", self.uniqe_classes)
        

    def __pre_train(self): 
        """Split X and Y into training and validation data
        """
        self.train_X, self.val_X, self.train_Y, self.val_Y = train_test_split(
            self.x, self.y, shuffle=True, test_size=0.1)
        if self.verbose:
            print("Training Data")
            print("Shape of train_X = %s and train_Y = %s" %
              (self.train_X.shape, self.train_Y.shape))
            print("Shape of val_X = %s and val_Y = %s" % (self.val_X.shape, self.val_Y.shape))
        

    def train(self, epochs, batch_size, model_name):
        """Traing the model

        Args:
            epochs (int): number of epochs
            batch_size (int): batch size
            model_name (str): model name
        """
        self.__pre_train()
        self.__create_model()
        self.model_name = model_name
        checkpoint = ModelCheckpoint(
            "checkpoint/cp-%s-{epoch:04d}.ckpt" % (model_name), monitor='val_acc', verbose=1, save_best_only=True, mode='min')

        self.model.fit(self.train_X, self.train_Y, epochs=epochs, batch_size=batch_size,
                       validation_data=(self.val_X, self.val_Y), callbacks=[checkpoint])
        self.save_model()

    def save_model(self):
        """Save model to a file
        """
        self.model.save("%s.h5" % (self.model_name))


    def __create_model(self):
        """Create model layers
        """
        self.model = Sequential()
        self.model.add(Embedding(self._vocab_size,
                                 128, input_length=self._max_len))
        self.model.add(Bidirectional(LSTM(128)))
        self.model.add(Dense(34, activation="relu"))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(len(self.uniqe_classes), activation="softmax"))
        self.model.compile(loss="categorical_crossentropy",
                    optimizer="adam", metrics=["accuracy"])
        self.model.summary()
        
    def load(self, path): 
        """Load model from a path 

        Args:
            path (str): model file path 
        """
        self.model = load_model(path)

    def __predictions(self, text):
        """Prediect function

        Args:
            text (str): input text

        Returns:
            list: list of predected classes number 
        """
        tokens = self.nlp.process(text).sentences[0].tokens
        encoded_tokens = self.x_token.texts_to_sequences(tokens)
        #Check for unknown words
        if [] in encoded_tokens:
            encoded_tokens = list(filter(None, encoded_tokens))

        encoded_tokens = np.array(encoded_tokens).reshape(1, len(encoded_tokens))

        x = pad_sequences(encoded_tokens, maxlen=self._max_len,
                          padding="post")

        pred = self.model.predict_proba(x)

        return pred

    def __get_final_output(self, pred):
        """match predected number with actual classes

        Args:
            pred (list): list of predected classes number 

        Returns:
            list: list of tuples ordered by most matching class
        """
        predictions = pred[0]

        classes = np.array(self.uniqe_classes)
        ids = np.argsort(-predictions)
        classes = classes[ids]
        predictions = -np.sort(-predictions)
        predicted_classes = []
        for i in range(pred.shape[1]):
            predicted_classes.append((classes[i], predictions[i]))
        return predicted_classes

    def classify(self, text):
        """Classify funtion

        Args:
            text (str): input text

        Returns:
            list: list of tuples ordered by most matching class
        """
        pred = self.__predictions(text)
        classes = self.__get_final_output(pred)
        return classes

    @property
    def uniqe_classes(self):
        """return uniqe classes as list

        Returns:
            list: list of uniqe classes 
        """
        return list(self._uniqe_classes)
