import re
from nltk.tokenize import sent_tokenize
import contractions
from datasets import load_dataset
import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
from keras import backend as K
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Concatenate, TimeDistributed, Attention 
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.text import Tokenizer 
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
K.clear_session()
gpu_devices = tf.config.list_physical_devices('GPU')

for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)


DATASET_LENGTH = 10000

max_text_len = 300
max_summary_len = 30

dataset = load_dataset("xsum")
ds = dataset['train'][:DATASET_LENGTH]

#Text Preprocessing
def clean_text(text, startend = False):
    clean_articles = []
    for article in text:
        expanded_words = []     #expand contractions
        count = 0
        for words in article.split():
            expanded_words.append(contractions.fix(words))
            count += 1
        article = " ".join(expanded_words)
        sent_list = sent_tokenize(article)
        clean_sent =[re.sub("\W+", " ",sent).lower() for sent in sent_list]  #removes punctuation and makes all words lower case
        if startend == True:
            clean_sent = ["sostok"] + clean_sent + ["eostok"]   
        clean_articles.append(" ".join(clean_sent))
    return clean_articles


# clean_doc = np.array(clean_text(ds['document'] ))
# clean_sum = np.array(clean_text(ds['summary'],True))
clean_doc = clean_text(ds['document'] )
clean_sum = clean_text(ds['summary'],True)

short_text = []
short_summary = []

for i in range(len(clean_doc)):
    if len(clean_sum[i].split()) <= max_summary_len and len(clean_doc[i].split()) <= max_text_len:
        short_text.append(clean_doc[i])
        short_summary.append(clean_sum[i])

clean_doc = np.array(short_text)
clean_sum = np.array(short_summary) 


x_train,x_test,y_train,y_test = train_test_split(clean_doc,clean_sum, test_size=0.2,shuffle=True)

x_tokenizer = Tokenizer() 
x_tokenizer.fit_on_texts(list(x_train))

thresh = 5

cnt = 0
tot_cnt = 0

for key, value in x_tokenizer.word_counts.items():
    tot_cnt = tot_cnt + 1
    if value < thresh:
        cnt = cnt + 1
print(tot_cnt)
print("% of rare words in vocabulary: ", (cnt / tot_cnt) * 100)



# Prepare a tokenizer
x_tokenizer = Tokenizer(num_words = tot_cnt - cnt)
x_tokenizer.fit_on_texts(list(x_train))

# Convert text sequences to integer sequences 
x_tr_seq = x_tokenizer.texts_to_sequences(x_train) 
x_val_seq = x_tokenizer.texts_to_sequences(x_test)

# Pad zero upto maximum length
x_tr = pad_sequences(x_tr_seq,  maxlen=max_text_len, padding='post')
x_val = pad_sequences(x_val_seq, maxlen=max_text_len, padding='post')

# Size of vocabulary (+1 for padding token)
x_voc = x_tokenizer.num_words + 1


y_tokenizer = Tokenizer()   
y_tokenizer.fit_on_texts(list(y_train))

thresh = 5

cnt = 0
tot_cnt = 0

for key, value in y_tokenizer.word_counts.items():
    tot_cnt = tot_cnt + 1
    if value < thresh:
        cnt = cnt + 1
    
print("% of rare words in vocabulary:",(cnt / tot_cnt) * 100)



y_tokenizer = Tokenizer(num_words = tot_cnt - cnt) 
y_tokenizer.fit_on_texts(list(y_train))


# Convert text sequences to integer sequences 
y_tr_seq = y_tokenizer.texts_to_sequences(y_train) 
y_val_seq = y_tokenizer.texts_to_sequences(y_test) 

# Pad zero upto maximum length
y_tr = pad_sequences(y_tr_seq, maxlen=max_summary_len, padding='post')
y_val = pad_sequences(y_val_seq, maxlen=max_summary_len, padding='post')

# Size of vocabulary (+1 for padding token)
y_voc = y_tokenizer.num_words + 1


# print("Size of vocabulary in X = {}".format(x_voc))
# print("Size of vocabulary in Y = {}".format(y_voc))

def unique_words(text):
    #Get all unique words
    unique_words = set()
    for i in text :
        for sent in i:
            for word in sent.split():
                unique_words.add(word)
    return unique_words
    
# def embedding_matrix():
#     #Get pretrained embeddings
#     embeddings_wv = gensim.models.KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin.gz", binary=True)
#     embeddings_wv.init_sims(replace=True)

#     unique_w = unique_words(x_train)

#     #Create our own embedding matrix with words in our articles   
#     wordIndex = {list(unique_w)[i]:i for i in range(len(list(unique_w)))}
#     vocab_size = len(unique_w)
#     embeddings = np.zeros((vocab_size,300))

#     #if word is in pretrained embedding, take that embedding and add it to our matrix
#     for word,index in wordIndex.items():
#         try:
#             embedding_vector = embeddings_wv[word]
#             if embedding_vector is not None:
#                 embeddings[index] = embedding_vector
#         except:
#             pass
#     return embeddings


class myModel():
    def __init__(self):
        self.latent_dim = 300
        self.embedding_dim= 200
        # Encoder
        self.encoder_inputs = Input(shape=(max_text_len,))

        #embedding layer
        self.enc_emb =  Embedding(x_voc, self.embedding_dim,trainable=True)(self.encoder_inputs)

        #encoder lstm 1
        self.encoder_lstm1 = LSTM(self.latent_dim,return_sequences=True,return_state=True, dropout = 0.4)

        #encoder lstm 2
        self.encoder_lstm2 = LSTM(self.latent_dim,return_sequences=True,return_state=True, dropout = 0.4)

        #encoder lstm 3
        self.encoder_lstm3= LSTM(self.latent_dim, return_state=True, return_sequences=True, dropout = 0.4)

        # Set up the decoder, using `encoder_states` as initial state.
        self.decoder_inputs = Input(shape=(None,))

        #embedding layer
        self.dec_emb_layer = Embedding(y_voc, self.embedding_dim,trainable=True)
        

        self.decoder_lstm = LSTM(self.latent_dim, return_sequences=True, return_state=True, dropout = 0.4)

        #Attention Layer
        self.attn_layer = Attention(name='attention_layer') 


        #dense layer
        self.decoder_dense =  TimeDistributed(Dense(y_voc, activation='softmax'))

    def createModel(self):
        encoder_output1, state_h1, state_c1 = self.encoder_lstm1(self.enc_emb)
        encoder_output2, state_h2, state_c2 = self.encoder_lstm2(encoder_output1)
        encoder_outputs, state_h, state_c= self.encoder_lstm3(encoder_output2)
        dec_emb = self.dec_emb_layer(self.decoder_inputs)
        decoder_outputs,decoder_fwd_state, decoder_back_state = self.decoder_lstm(dec_emb,initial_state=[state_h, state_c])
        attn_out, attn_states = self.attn_layer([encoder_outputs, decoder_outputs]) 
        # Concat attention output and decoder LSTM output 
        decoder_concat_input = Concatenate(axis=-1, name='concat_layer')([decoder_outputs, attn_out])
        decoder_outputs = self.decoder_dense(decoder_outputs)
        self.model = Model([self.encoder_inputs, self.decoder_inputs], decoder_outputs)
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
        
        # Encode the input sequence to get the feature vector
        self.encoder_model = Model(inputs=self.encoder_inputs,outputs=[encoder_outputs, state_h, state_c])
        # Decoder setup
        # Below tensors will hold the states of the previous time step
        decoder_state_input_h = Input(shape=(self.latent_dim,))
        decoder_state_input_c = Input(shape=(self.latent_dim,))
        decoder_hidden_state_input = Input(shape=(max_text_len,self.latent_dim))

        # Get the embeddings of the decoder sequence
        dec_emb2= self.dec_emb_layer(self.decoder_inputs) 
        
        # To predict the next word in the sequence, set the initial states to the states from the previous time step
        decoder_outputs2, state_h2, state_c2 = self.decoder_lstm(dec_emb2, initial_state=[decoder_state_input_h, decoder_state_input_c])

        # A dense softmax layer to generate prob dist. over the target vocabulary
        decoder_outputs2 = self.decoder_dense(decoder_outputs2) 
        self.decoder_model = Model([self.decoder_inputs] + [decoder_hidden_state_input,decoder_state_input_h, decoder_state_input_c],
        [decoder_outputs2] + [state_h2, state_c2])

        return self.model
    
    def train(self):
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience=2)
        history=self.model.fit([x_tr,y_tr[:,:-1]], y_tr.reshape(y_tr.shape[0],y_tr.shape[1], 1)[:,1:] ,epochs=50,callbacks=[es],batch_size=128, validation_data=([x_val,y_val[:,:-1]], y_val.reshape(y_val.shape[0],y_val.shape[1], 1)[:,1:]))

    def decode_sequence(self,input_seq):
        # Encode the input as state vectors.
        reverse_target_word_index=y_tokenizer.index_word
        target_word_index=y_tokenizer.word_index
        e_out, e_h, e_c = self.encoder_model.predict(input_seq)
        
        # Generate empty target sequence of length 1.
        target_seq = np.zeros((1,1))
        
        # Populate the first word of target sequence with the start word.
        target_seq[0, 0] = target_word_index['sostok']

        stop_condition = False
        decoded_sentence = ''
        while not stop_condition:
        
            output_tokens, h, c = self.decoder_model.predict([target_seq] + [e_out, e_h, e_c])

            # Sample a token
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_token = reverse_target_word_index[sampled_token_index]
            
            if(sampled_token!='eostok'):
                decoded_sentence += ' '+sampled_token

            # Exit condition: either hit max length or find stop word.
            if (sampled_token == 'eostok'  or len(decoded_sentence.split()) >= (max_summary_len-1)):
                stop_condition = True

            # Update the target sequence (of length 1).
            target_seq = np.zeros((1,1))
            target_seq[0, 0] = sampled_token_index

            # Update internal states
            e_h, e_c = h, c

        return decoded_sentence

    def predict(self, x_test,y_test,x_val):
        for i in range(100):
            print("Review:",(x_test[i]))
            print("Original summary:",(y_test[i]))
            print("Predicted summary:",self.decode_sequence(x_val[i].reshape(1,max_text_len)))
            print("\n")


testModel = myModel()
testModel.createModel()
testModel.train()
testModel.predict(x_test,y_test,x_val)










