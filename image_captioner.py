import pandas as pd
import numpy as np
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.models import Model,Sequential
from keras.preprocessing import image
from keras.preprocessing.sequence import pad_sequences
import string
from keras.utils import to_categorical
from keras.layers import Input,Dropout,Dense,LSTM,Embedding,ZeroPadding2D,Conv2D,MaxPooling2D,Flatten
from keras.layers.merge import add
from keras.callbacks import ModelCheckpoint
import os
import pickle
from keras.preprocessing.text import Tokenizer
from feature_extractor import feature_extractor
class Captioner_model:
    def __init__(self):
        pass
    def define_model(self,max_length,vocab_size):
        inp1 = Input(shape=(4096,))
        im1 = Dropout(0.5)(inp1)
        im2 = Dense(256,activation='relu')(im1)
        im3=Dense(256,activation='relu')(im2)
        inp2 = Input(shape=(max_length,))
        tx1 = Embedding(vocab_size, 256, mask_zero=True)(inp2)
        tx2 = Dropout(0.5)(tx1)
        tx3 = LSTM(256)(tx2)
        comb1 = add([im3, tx3])
        comb2 = Dense(256, activation='relu')(comb1)
        comb3 = Dense(256, activation='relu')(comb2)
        oups = Dense(vocab_size, activation='softmax')(comb3)
        model = Model(inputs=[inp1, inp2], outputs=oups)
        model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
        #print(model.summary())
        return model
def VGG_16():
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(224,224,3)))
    model.add(Conv2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    return model



class image_captioner:
    def __init__(self):
        self.max_length=34
        self.vocab_size=8766
        with open('tokenizer.pkl', 'rb') as f:
            self.tokenizer=pickle.load(f)
        cap=Captioner_model()
        self.model=cap.define_model(self.max_length,self.vocab_size)
        self.model.load_weights('image_captioner_weights.h5')
        # with open('vgg_model.pkl', 'rb') as f:
        #     self.vgg_model=pickle.load(f)
        # self.vgg_model=VGG16()
        # self.vgg_model.load_weights('vgg_model16.h5')
        # self.vgg_model=VGG16()
        # self.vgg_model.layers.pop()
        # self.vgg_model=Model(inputs=self.vgg_model.inputs,outputs=self.vgg_model.layers[-1].output)
    def int_to_word(self,integer):
        for word, index in self.tokenizer.word_index.items():
            if index == integer:
                return word
        return None
    # def feature_extractor(self,imgg):
    #     img=image.img_to_array(imgg)
    #     img=img.reshape((1,img.shape[0],img.shape[1],img.shape[2]))
    #     img=preprocess_input(img)
    #     features=self.vgg_model.predict(img)
    #     return features
    def generate_desc(self,picture):
        photo=feature_extractor(picture)
        text='startseq'
        for i in range(self.max_length):
            seq=self.tokenizer.texts_to_sequences([text])[0]
            seq=pad_sequences([seq],maxlen=self.max_length)
            yhat=self.model.predict([photo,seq])
            yhat=np.argmax(yhat)
            word=self.int_to_word(yhat)
            if word=='endseq':
                break
            if word is None:
                break
            text=text + ' ' + word
        return text[9:]
    