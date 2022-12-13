#bi-lstm model
import numpy as np
import pandas as pd
import sys
import tensorflow as ts
import re
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Bidirectional, GlobalMaxPool1D, Dense, LSTM, Conv1D, Embedding
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import pickle
from gensim.models import Word2Vec
import seaborn as sns
from wordcloud import WordCloud
import matplotlib.pyplot as plt
def getModel():
    embedding_layer = Embedding(input_dim = vocab_length,
                                output_dim = Embedding_dimensions,
                                weights=[embedding_matrix],
                                input_length=input_length,
                                trainable=False)

    model = Sequential([
        embedding_layer,
        Bidirectional(LSTM(100, dropout=0.8, return_sequences=True)),
        Bidirectional(LSTM(100, dropout=0.5, return_sequences=True)),
        Bidirectional(LSTM(100, dropout=0.5, return_sequences=True)),
        Bidirectional(LSTM(100, dropout=0.3, return_sequences=True)),
        Conv1D(100, 5, activation='relu'),
        GlobalMaxPool1D(),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid'),
    ],
    name="Sentiment_Model")
    return model

filename = sys.argv[1] + ".tsv"
df = pd.read_csv(filename, header=None, sep="\t")   # reading into dataframe
tsv_file = df.values 
random.shuffle(tsv_file) #
sentences = []
pos = []
neg = []
scores = []
ns = []
ps = []

for line in tsv_file:
    sentences.append(line[0])
    scores.append(line[1])
    if(line[1]==0):
        neg.append(line[0])
        ns.append(line[1])
    else:
        pos.append(line[0])
        ps.append(line[1])
max_val = min(len(pos), len(neg))
pos = pos[:max_val]
neg = neg[:max_val]
ps = ps[:max_val]
ns = ns[:max_val]
sentences2 = pos + neg
scores2 = ps+ns
print("Positives: ", len(pos))
print("Negatives: ", len(neg))


wc = WordCloud(max_words = 1000 , width = 1600 , height = 800,
              collocations=False, background_color="white").generate(" ".join(pos))
wc.to_file(sys.argv[1]+"_pos_words.png")
wc = WordCloud(max_words = 1000 , width = 1600 , height = 800,
              collocations=False, background_color="white").generate(" ".join(neg))
wc.to_file(sys.argv[1]+"_neg_words.png")
X_data, y_data = np.array(sentences2), np.array(scores2)

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data,
                                                    test_size = 0.05, random_state = 0)
print('Data Split done.')

#word embeddings
Embedding_dimensions = 100
input_length = 60 
vocab_length = 60000

Word2vec_train_data = list(map(lambda x: x.split(), X_train))
word2vec_model = Word2Vec(Word2vec_train_data,
                 vector_size=Embedding_dimensions,
                 workers=8,
                 min_count=5)

print("Vocabulary Length:", len(word2vec_model.wv.key_to_index))
word2vec_model.save("saved_models/embeddings/"+sys.argv[1]+".bin")

tokenizer = Tokenizer(filters="", lower=False, oov_token="<oov>")
tokenizer.fit_on_texts(X_data)
tokenizer.num_words = vocab_length
print("Tokenizer vocab length:", vocab_length)

X_train = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=input_length)
X_test  = pad_sequences(tokenizer.texts_to_sequences(X_test) , maxlen=input_length)
labelencoder = LabelEncoder()
y_train = labelencoder.fit_transform(y_train)
y_test=labelencoder.fit_transform(y_test)
print("X_train.shape:", X_train.shape)
print("X_test.shape :", X_test.shape)

embedding_matrix = np.zeros((vocab_length, Embedding_dimensions))

for word, token in tokenizer.word_index.items():
    if word2vec_model.wv.__contains__(word):
        if token < 6000:
            embedding_matrix[token] = word2vec_model.wv.__getitem__(word)
print("Embedding Matrix Shape:", embedding_matrix.shape)

training_model = getModel()
training_model.summary()

callbacks = [ReduceLROnPlateau(monitor='val_loss', patience=5, cooldown=0),
             EarlyStopping(monitor='val_accuracy', min_delta=1e-4, patience=5)]

training_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model_history = training_model.fit(
    X_train, y_train,
    batch_size=1024,
    epochs=100,
    validation_split=0.1,
    callbacks=callbacks,
    verbose=0,
)

acc = model_history.history['accuracy']
val_acc = model_history.history['val_accuracy']
loss = model_history.history['loss']
val_loss = model_history.history['val_loss']
epochs=range(len(acc))
#Ooutput for inspection
plt.plot(epochs,acc,label='Trainin_acc',color='blue')
plt.plot(epochs,val_acc,label='Validation_acc',color='red')
plt.legend()
plt.title("Training and Validation Accuracy")
plot_name = sys.argv[1]+"val_acc.png"
plt.savefig(plot_name)
training_model.save("saved_models/"+sys.argv[1])
print("Test data performance")
scores = training_model.predict(X_test, verbose=1, batch_size=1024)
y_pred=np.where(scores>0.5,1,0)
print("Confusion Matrix")
cm=confusion_matrix(y_pred,y_test)
print(cm)
print("Accuracy")
print(accuracy_score(y_pred,y_test))

print(classification_report(y_test, y_pred))
