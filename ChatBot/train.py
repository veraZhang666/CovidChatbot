
import spacy # 3.x version
from spacy import displacy
import re # regular expressions
#import en_core_web_lg # Large SpaCy model for English language
import numpy as np
from collections import defaultdict
import pandas as pd
import torch
#load question and intent dataset
data_df = pd.read_csv('covid_19.csv',names=["questions","intent"])

response_df = pd.read_csv('response.csv',names=["intent","response"])
responses_dict = response_df.set_index('intent').T.to_dict('list')
print(responses_dict)

nlp = spacy.load("en_core_web_lg")
def get_all_tokens(text):
    doc = nlp(text)
    d = defaultdict(list)
    print(doc)
    tokens = []
    for token in doc:
        print(token.text, token.pos_, token.dep_) #(str,str,str)
        tokens.append(token.text)
    return tokens
get_all_tokens('I did my homework at 2 pm in the afternoon ')


def get_all_entities(text):
    """
        Get all entities in a given text, in a text: label_ dictionary
    """
    doc = nlp(text)
    # print(doc)

    d = defaultdict(list)
    for ent in doc.ents:
        d[ent.label_].append(ent.text)
    print(d)
    return (d)


text = 'I want to travel to Beijing with LiLi at 3:00 pm in May 1st of 2021'
test_ents = get_all_entities(text)

# Generate data and time entities, we can put them in the answer
def CheckDataTimeEnts(entities):
    time = ""
    if 'TIME' in entities and 'DATE' in entities:
        time =  ' '.join(entities['TIME']) +' in '+ ' '.join(entities['DATE'])
    elif 'TIME' in entities:
        time = ' and '.join(entities['TIME'])
    elif 'DATE' in entities:
        time = ' and '.join(entities['DATE'])
    else:
        time = ""
    return time
CheckDataTimeEnts(test_ents)

#calculate the similarity
doc1 = nlp(u"what is the symtom of crona virus")
doc2 = nlp(u"contracted the crona virus")
doc3 = nlp(u'is fever one of the symtom of HPV?')
similarity = doc1.similarity(doc2)
s2 = doc2.similarity(doc3)
print(similarity)
print(s2)

trainining_sentences =data_df.questions

#conver the intent to one hot code
training_intents =  pd.get_dummies(data_df.intent)



# Creating the word embedding vectors shape:(sentence_len,300)
# 1 sentence map to 300 dimention vector.
def getWordVectors(trainining_sentences):
    embed_vec_dim = nlp('').vocab.vectors_length  # 300 dim
    X_train = np.zeros((len(trainining_sentences), embed_vec_dim))

    print('training data shape : ', X_train.shape)

    for i, sentence in enumerate(trainining_sentences):
        # lower the sentence
        sentence = sentence.lower()
        # Pass each each sentence to the nlp object to create a document
        doc = nlp(sentence)
        # Save the document's .vector attribute to the corresponding row in X
        X_train[i, :] = doc.vector
    return X_train

X_train = getWordVectors(trainining_sentences)

y_labels = []

intent2label = {'greetings':0,'information':1,'prevention':2,
                'symptoms':3, 'travel':4,'vaccine' :5}

label_tensors = torch.zeros(len(trainining_sentences),1)

for intent in intent2label.keys():
    l = training_intents.loc[training_intents[intent] == 1].index.tolist()
    for idx in l :
        label_tensors[idx][0]=intent2label[intent]


X_train = torch.from_numpy(X_train)
y_train = label_tensors
# ===============================dataloader =====================


from torch.utils.data import DataLoader,Dataset
import torch.nn as nn

dataset = Dataset()


class MyDataset(Dataset):
    def __init__(self,X,y):
        X = X_train
        y = y_train
        self.X = X
        self.y = y
    def __getitem__(self, idx):
        return self.X[idx],self.y[idx]
    def __len__(self):
        return self.X.shape[0]
dataset  = MyDataset(X_train,y_train)

dataloader = DataLoader(dataset = dataset, shuffle = True, batch_size = 1 )


#===========================Modeling ========================



torch.set_default_tensor_type(torch.DoubleTensor)

NUM_EPOCHS = 300

fc=torch.nn.Linear(300,6) #只使用一层线性分类器
fc = fc.double()

criterion=torch.nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(fc.parameters())

for epoch in range(NUM_EPOCHS):
    for idx, (images,labels) in enumerate(dataloader):
        x =images.reshape(-1,300)
        labels = labels.squeeze(1).long()
        optimizer.zero_grad() #梯度清零
        preds=fc(x) #计算预测

        loss=criterion(preds,labels) #计算损失
        loss.backward() # 计算参数梯度
        optimizer.step() # 更新迭代梯度
        if epoch % 50 ==0:
            if idx % 20 ==0:
                print('epoch={}:idx={},loss={:g}'.format(epoch,idx,loss))

correct=0
total=0

for idx,(images,labels) in enumerate(dataloader):
    x =images.reshape(-1,300) #对所有的图片进行reshape size(m,28*28)

    preds=fc(x)
    predicted=torch.argmax(preds,dim=1) #在dim=1中选取max值的索引
    if idx ==0:
        print('x size:{}'.format(x.size()))
        print('preds size:{}'.format(preds.size()))
        print('predicted size:{}'.format(predicted.size()))

    total+=labels.size(0)
    correct+=(predicted == labels).sum().item()
    #print('##########################\nidx:{}\npreds:{}\nactual:{}\n##########################\n'.format(idx,predicted,labels))

accuracy=correct/total
print('{:1%}'.format(accuracy))

torch.save(fc.state_dict(),'classify_model.pth')
