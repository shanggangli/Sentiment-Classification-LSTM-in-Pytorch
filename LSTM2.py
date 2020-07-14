#-*- codeing=utf-8 -*-
#@time: 2020/7/13 20:21
#@Author: Shang-gang Lee
import numpy as np                # deal with data
import pandas as pd               # deal with data
import re                         # regular expression
from bs4 import BeautifulSoup     # resolver review
from nltk.corpus import stopwords # Import the stop word list
from gensim.models import word2vec# use word2Vec(skip-gram model) making wordfeature vetor
from sklearn.model_selection import train_test_split # use trian data split train and test data
import torch
from torch.utils.data import Dataset,TensorDataset
import torch.nn as nn


train_data=pd.read_table(r'C:\Users\朝花夕拾\Desktop\机器学习\kaggle\word2vec-nlp-tutorial\labeledTrainData.tsv',header=0,delimiter="\t", quoting=3)
test_data=pd.read_table(r'C:\Users\朝花夕拾\Desktop\机器学习\kaggle\word2vec-nlp-tutorial\testData.tsv',header=0,delimiter="\t", quoting=3)
unlabeled_train_data=pd.read_table(r'C:\Users\朝花夕拾\Desktop\机器学习\kaggle\word2vec-nlp-tutorial\unlabeledTrainData.tsv',header=0,delimiter="\t", quoting=3,error_bad_lines=False)

def review_to_wordlist(review,remove_stop_words=False):
    #1.remove HIML
    reivew_text=BeautifulSoup(review,'lxml').get_text()

    #2.Remove non-latters
    latters_only=re.sub("[^a-zA-Z]",' ',reivew_text)

    #3.Convert to lower case,split into individual words
    words=latters_only.lower().split()

    #4.Remove stop words
    if remove_stop_words:
        stop=set(stopwords.words('english'))
        words=[w for w in words if not w in stop]

    #5. reutrn a list of words
    return words

#test review to wordlist
word=[]
n=train_data['review']
for review in n:
    #print(review)
    word.append(review_to_wordlist(review,remove_stop_words=True))
#print(word)

#word2vec model
# Set values for various parameters
num_features = 300    # Word vector dimensionality
min_word_count = 5   # Minimum word count
num_workers = 4       # Number of threads to run in parallel
context = 10          # Context window size
downsampling = 1e-3   # Downsample setting for frequent words
model=word2vec.Word2Vec(word,workers=num_workers,size=num_features,
                        min_count=min_word_count,window=context,sample=downsampling)

# make features vector by each words
def makeFeatureVec(words,model,num_features):
    featureVec = np.zeros((num_features,), dtype="float32")
    nwords=0
    index2word_set = set(model.wv.index2word)    #get name
    for word in words:
        if word in index2word_set:               # if word in index2word and get it's feature
            nwords+=1
            featureVec=np.add(featureVec,model[word])
    featureVec=np.divide(featureVec,nwords)      # average each featureVector
    return featureVec

# make all word's features
def getAvgFeatureVecs(reviews,model,num_features):
    counter=0
    reviewFeatureVecs = np.zeros((len(reviews), num_features), dtype="float32") #features size=len(reviews) X num_features
    for review in reviews:                                                      # loop each review(word)
        reviewFeatureVecs[counter]=makeFeatureVec(review,model,num_features)    # get each word's featureVectors
        counter+=1
    return reviewFeatureVecs

#train features
getAvgFeatureVecs_train=[]
getAvgFeatureVecs_train.append(getAvgFeatureVecs(reviews=word,model=model,num_features=300)) #get WordFeatureVetor into list

# build LSTM
class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.lstm=nn.LSTM(input_size=300,           # input_size=word'featureVectors=300
                          hidden_size=256,          # hidden_size=256
                          num_layers=1,             # num of running layer
                          batch_first=True)         # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        self.fc=nn.Linear(256,2)                    #full conntion

    def forward(self,x):
        # x shape (batch, time_step, input_size)
        # out shape (batch, time_step, output_size)
        # hidden shape (n_layers, batch, hidden_size)
        # cell shape (n_layers, batch, hidden_size)
        out,(hidden,cell)=self.lstm(x)
        out=self.fc(out[:,-1,:])                    # use the last time_step to input full conntion
        out=torch.sigmoid(out)                      # make output in (0,1)
        return out
lstm=LSTM()

# trian data
getAvgFeatureVecs_train=torch.from_numpy(np.array(getAvgFeatureVecs_train))         # array->torch
getAvgFeatureVecs_train=getAvgFeatureVecs_train.view(-1,1,300)                      # (num of sentence,batch,word_featureVetor)
y_label=torch.from_numpy(np.array(train_data['sentiment']))                         # get label
#print(y_label)
y_label=y_label.view(-1,1)                                                          #(num of sentence,label)
# print(getAvgFeatureVecs_train.shape)
# # print(y_train.shape)

X_train,X_test,y_train,y_test=train_test_split(getAvgFeatureVecs_train,y_label,random_state=0,train_size=0.8) # get train data and test
# print(",训练数据特征:",X_train.shape,
#       ",测试数据特征:",X_test.shape)
# print(",训练数据标签:",y_train.shape,
#      ',测试数据标签:',y_test.shape )
y_test=np.array(y_test.view(-1))

#train data
#loading data
BATCH_SIZE=64               # batch size=64
Epoch=1                     # use epoch=1 for saving time and calculated amount
LR=0.01                     # learning rate
deal_traindata=TensorDataset(X_train,y_train)           # deal with wordVetor and label
load_train=torch.utils.data.DataLoader(dataset=deal_traindata, batch_size=BATCH_SIZE, shuffle=True)     #laod data make batch

#loss function
optimizer = torch.optim.Adam(lstm.parameters(), lr=LR)   # optimize all cnn parameters
loss_func=nn.CrossEntropyLoss()                          # loss function is CrossEntropyLoss

for epoch in range(Epoch):
    for step ,(x,label) in enumerate(load_train):
        # print(x.shape)
        #print(label.shape)
        label=label.view(-1)                            # loss function need 1 dim! if don't do it, loss function will make error!
        output=lstm(x)
        #print(output.shape)
        optimizer.zero_grad()                           # clear gradients for this training step
        loss=loss_func(output,label)
        loss.backward()                                 # backpropagation, compute gradients
        optimizer.step()                                # apply gradients
        if step%50==0:
            output_test=lstm(X_test)                    # test model and print loss and accuracy
            pred_y = torch.max(output_test, 1)[1].data.numpy()
            # print(pred_y.shape)
            # print(y_test.shape)
            accuracy = float((pred_y == y_test).astype(int).sum()) / float(y_test.size)
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)
