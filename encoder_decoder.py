import re
from nltk.tokenize import sent_tokenize
import contractions
from datasets import load_dataset
import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
import gensim
from keras import backend as K
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Concatenate, TimeDistributed, Bidirectional
from attention import AttentionLayer
DATASET_LENGTH = 100
dataset = load_dataset("multi_news")
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

#Text Preprocessing
def parse_article(article, startend = False):
    expanded_words = []     #expand contractions
    for words in article.split():
        expanded_words.append(contractions.fix(words))
    article = " ".join(expanded_words)
    sent_list = sent_tokenize(article)
    clean_sent =[re.sub("\W+", " ",sent).lower() for sent in sent_list]   #removes punctuation and makes all words lower case
    if startend == True:
        clean_sent = ["article_start"] + clean_sent + ["article_end"]   #removes punctuation and makes all words lower case with special tags article_start and article_end 
        
    return clean_sent

ds = dataset['train'][:DATASET_LENGTH]

clean_doc = [parse_article(text, True) for text in ds['document'] ]
clean_sum = [parse_article(text, True) for text in ds['summary'] ]
x_train,x_test,y_train,y_test = train_test_split(clean_doc,clean_sum, test_size=0.2,shuffle=True)

def unique_words(text):
    #Get all unique words
    unique_words = set()
    for i in text :
        for sent in i:
            for word in sent.split():
                unique_words.add(word)
    return unique_words
    
def embedding_matrix():
    #Get pretrained embeddings
    embeddings_wv = gensim.models.KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin.gz", binary=True)
    embeddings_wv.init_sims(replace=True)

    unique_w = unique_words(x_train)

    #Create our own embedding matrix with words in our articles   
    wordIndex = {list(unique_w)[i]:i for i in range(len(list(unique_w)))}
    vocab_size = len(unique_w)
    embeddings = np.zeros((vocab_size,300))

    #if word is in pretrained embedding, take that embedding and add it to our matrix
    for word,index in wordIndex.items():
        try:
            embedding_vector = embeddings_wv[word]
            if embedding_vector is not None:
                embeddings[index] = embedding_vector
        except:
            pass
    return embeddings

max_len_text = 80
max_len_summary = 10
embeddings = embedding_matrix()
latent_dim = 500
x_voc = unique_words(x_train)
y_voc = unique_words(y_train)
x_voc_size = len(x_voc)
y_voc_size = len(y_voc)
reverse_source_word_index = {i:list(x_voc)[i] for i in range(len(list(x_voc)))}
reverse_target_word_index = {i:list(y_voc)[i] for i in range(len(list(y_voc)))}
target_word_index = {list(y_voc)[i]:i for i in range(len(list(y_voc)))}

#encoder
encoder_inputs = Input(shape=(max_len_text,)) 
enc_emb = Embedding(x_voc_size,300,input_length=max_len_text,weights=[embeddings],trainable=False)(encoder_inputs) 

#LSTM 1
encoder_lstm1 = LSTM(latent_dim,return_sequences=True,return_state=True,dropout=0.4,recurrent_dropout=0.4) # mess with dropout values to deal with over fitting
encoder_output1, state_h1, state_c1 = encoder_lstm1(enc_emb) 

#LSTM 2 
encoder_lstm2 = LSTM(latent_dim,return_sequences=True,return_state=True,dropout=0.4,recurrent_dropout=0.4) 
encoder_output2, state_h2, state_c2 = encoder_lstm2(encoder_output1) 

#LSTM 3 
encoder_lstm3=LSTM(latent_dim, return_state=True, return_sequences=True,dropout=0.4,recurrent_dropout=0.4) 
encoder_outputs, state_h, state_c= encoder_lstm3(encoder_output2) 

# Set up the decoder. 
decoder_inputs = Input(shape=(None,)) 
dec_emb_layer = Embedding(x_voc_size,300,input_length=max_len_text,weights=[embeddings],trainable=False,) 

dec_emb = dec_emb_layer(decoder_inputs)

decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True,dropout=0.4,recurrent_dropout=0.25) 
decoder_outputs,decoder_fwd_state, decoder_back_state = decoder_lstm(dec_emb,initial_state=[state_h, state_c])

attn_layer = AttentionLayer(name='attention_layer') 
attn_out, attn_states = attn_layer([encoder_outputs, decoder_outputs])

# Concat attention output and decoder LSTM output 
decoder_concat_input = Concatenate(axis=-1, name='concat_layer')([decoder_outputs, attn_out])

#Dense layer
decoder_dense = TimeDistributed(Dense(y_voc_size, activation='softmax')) 
decoder_outputs = decoder_dense(decoder_concat_input) 

# Define the model
model = Model([encoder_inputs, decoder_inputs], decoder_outputs) 
model.summary()

model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy')

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience=3)
checkpoint = ModelCheckpoint('model_best_weights_news.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min', period=1)

history=model.fit([x_train,y_train[:,:-1]], y_train.reshape(y_train.shape[0],y_train.shape[1], 1)[:,1:] ,epochs=100,batch_size=128, validation_data=([x_test,y_test[:,:-1]], y_test.reshape(y_test.shape[0],y_test.shape[1], 1)[:,1:]),callbacks = [es,checkpoint])

# print(parse_reviews(dataset['train'][1]['document']))
# print(dataset['train'][1]['document'])
encoder_model = Model(inputs=encoder_inputs,outputs=[encoder_outputs, state_h, state_c])

# decoder inference
# Below tensors will hold the states of the previous time step
decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_hidden_state_input = Input(shape=(max_len_text,latent_dim))

# Get the embeddings of the decoder sequence
dec_emb2= dec_emb_layer(decoder_inputs)

# To predict the next word in the sequence, set the initial states to the states from the previous time step
decoder_outputs2, state_h2, state_c2 = decoder_lstm(dec_emb2, initial_state=[decoder_state_input_h, decoder_state_input_c])

#attention inference
attn_out_inf, attn_states_inf = attn_layer([decoder_hidden_state_input, decoder_outputs2])
decoder_inf_concat = Concatenate(axis=-1, name='concat')([decoder_outputs2, attn_out_inf])

# A dense softmax layer to generate prob dist. over the target vocabulary
decoder_outputs2 = decoder_dense(decoder_inf_concat)

# Final decoder model
decoder_model = Model(
[decoder_inputs] + [decoder_hidden_state_input,decoder_state_input_h, decoder_state_input_c],
[decoder_outputs2] + [state_h2, state_c2])


def decode_sequence(input_sequence):
    # Encode the input as state vectors.
    e_out, e_h, e_c = encoder_model.predict(input_sequence)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1,1))

    # Chose the 'start' word as the first word of the target sequence
    target_seq[0, 0] = target_word_index['article_start']

    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + [e_out, e_h, e_c])

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_token = reverse_target_word_index[sampled_token_index]

        if(sampled_token!='end'):
            decoded_sentence += ' '+sampled_token

            # Exit condition: either hit max length or find stop word.
            if (sampled_token == 'article_end' or len(decoded_sentence.split()) >= (max_len_summary-1)):
                stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1,1))
        target_seq[0, 0] = sampled_token_index

        # Update internal states
        e_h, e_c = h, c

    return decoded_sentence

def seq2summary(input_seq):
    newString=''
    for i in input_seq:
      if((i!=0 and i!=target_word_index['article_start']) and i!=target_word_index['article_end']):
        newString=newString+reverse_target_word_index[i]+' '
    return newString

def seq2text(input_seq):
    newString=''
    for i in input_seq:
      if(i!=0):
        newString=newString+reverse_source_word_index[i]+' '
    return newString

for i in range(100):
  print("Review:",seq2text(x_test[i]))
  print("Original summary:",seq2summary(y_test[i]))
  print("Predicted summary:",decode_sequence(x_test[i].reshape(1,max_len_text)))
  print("\n")