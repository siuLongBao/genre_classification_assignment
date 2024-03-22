import json;
import numpy as np;
import string;
from nltk import TreebankWordTokenizer;
import gensim;
from gensim.models import Word2Vec;
from gensim.models import KeyedVectors;
import torch;
import torch.nn as nn;
import torch.optim as optim;
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence;
from torch.utils.data import DataLoader;
from sklearn.metrics import confusion_matrix, f1_score, ConfusionMatrixDisplay;
import matplotlib.pyplot as plt;
import nltk;


#Loading word2Vec Model
"""
Remember to Remove Limit Before Training The Whole Model

Also can try to train a word2vec model for not included words (probably not necessary).
Spend time on hyperparameters tuning and transformers.
"""
path = 'C:/Users/user/Documents/Gensim Library/GoogleNews-vectors-negative300.bin.gz';
model = KeyedVectors.load_word2vec_format(path, binary=True, limit=50000);
word_included = model.key_to_index;


#Initializing TreebankWordTokenizer
tokenizer = TreebankWordTokenizer();

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu');


# load the training data
train_data = json.load(open("genre_train.json", "r"))
X = train_data['X']
Y = train_data['Y'] # id mapping: 0 - Horror, 1 - Science Fiction, 2 - Humour, 3 - Crime Fiction
docid = train_data['docid'] # these are the ids of the books which each training example came from

# load the test data
# the test data does not have labels, our model needs to generate these
test_data = json.load(open("genre_test.json", "r"))
Xt = test_data['X']
#print(Xt[0]);


# Spliting training data into training and validation data
train_val_number = int(len(X)*0.7);
X_training_data = X[0:train_val_number];
X_validation_data = X[train_val_number:];
Y_training_data = Y[:train_val_number];
Y_validation_data = Y[train_val_number:];



# Calculating class weights
count_frequency = {};
for i in Y:
    if (i in count_frequency):
        count_frequency[i] += 1;
    else:
        count_frequency[i] = 1;
print(count_frequency);
print(len(Y));
keys = sorted(count_frequency.keys());
class_weights = [];
for key in keys:
    class_weights.append(1-(count_frequency[key]/len(Y)));

class_weights = torch.tensor(class_weights);
print(class_weights);

#yjk;






"""
print("training X length: " + str(len(X_training_data)));
print(X_training_data[0]);
print("validation data length: " + str(len(X_validation_data)));
print(X_validation_data[0]);

print("Hello\n\n\n\n\n\n\nHello");
print(type(Y_validation_data[0]));
print(Y_validation_data[0]);
print(type(Y_validation_data));
"""

# Setting stopwords library
stopwords = set(nltk.corpus.stopwords.words("english"))


# Converting training Xs into tokens
X_train_tokens_list = [];
train_data_index = 0;
while train_data_index < len(X_training_data):
    tokens_list = tokenizer.tokenize(X_training_data[train_data_index]);
    data_token_list = [];
    for token in tokens_list:
        if token in string.punctuation or token in stopwords:
            continue;
        else:
            data_token_list.append(token);
    X_train_tokens_list.append(data_token_list);
    train_data_index += 1;
    
X_validation_tokens_list = [];
valid_data_index = 0;
while valid_data_index < len(X_validation_data):
    tokens_list = tokenizer.tokenize(X_validation_data[valid_data_index]);
    data_token_list = [];
    for token in tokens_list:
        if token in string.punctuation or token in stopwords:
            continue;
        else:
            data_token_list.append(token);
    X_validation_tokens_list.append(data_token_list);
    valid_data_index += 1;

"""
print("training X tokens length: " + str(len(X_train_tokens_list)));
print(X_train_tokens_list[0]);
print("validation data tokens length: " + str(len(X_validation_tokens_list)));
print(X_validation_tokens_list[0]);
"""
"""
# What is this data for???
print("1044: ",Xt[1044]);
print(Xt[1926]);
print(Xt[4527]);
"""


# Converting data into embedding
# Takes tokens_lists as input, an element in the X_train_tokens_list or X_validation_tokens_list
# Output embeddings_list and list_of_words_not_in_word2vec
# Each embeddings have a shape of (300,)
def convertToEmbedding(tokens_list):
    embedding_list = [];
    loss = [];
    for token in tokens_list:
        if token in word_included:
            embedding_list.append(model.get_vector(token));
        else:
            loss.append(token);
    return embedding_list, loss;


#Test convertToEmbedding method
embedding, loss = convertToEmbedding(X_train_tokens_list[0]);
#print(loss);

"""
keyList = list(word_included.keys());
print(type(keyList));
for i in range(100):
    print(model.get_vector(keyList[i]).shape);
"""


#Create Batch
def createBatchOfData(batch_size, data_bank):
    batch_of_data = [];
    if (len(data_bank)<=batch_size):
        batch_of_data = data_bank.copy();
        data_bank = [];
    else:
        for i in range(batch_size):
            random_num = np.random.randint(len(data_bank));
            data_to_add = data_bank[random_num].copy();
            del data_bank[random_num];
            batch_of_data.append(data_to_add);
    return batch_of_data, data_bank;


# Test createBatchOfData method
#print(len(X_train_tokens_list));
#batch, X_train_tokens_list = createBatchOfData(32, X_train_tokens_list);
#print(len(batch));
#print(batch[0]);
#print(len(X_train_tokens_list));
    


class DBPediaData(torch.utils.data.Dataset):
    def __init__(self, list_of_tokens_list, labels):
        # the titles in the data (our unprocessed X values)
        self.list_of_tokens_list = list_of_tokens_list; 
        # the entity types in the data (our y)
        self.labels = labels
    
    def __len__(self):
        return len(self.list_of_tokens_list);
    
    def __getitem__(self, index):
        embeddings_list, _ = convertToEmbedding(self.list_of_tokens_list[index]);
        #Note: we also return the length of the title to help with padding
        return embeddings_list, self.labels[index], len(embeddings_list);
    

# takes a list of data points provided by DBPediaData.__getitem__ and returns a
# batch that is ready for passing to the neural network model
def collate_dbpedia(batch):
    #print(batch[0][0]);
    # each title might be a different length so we must pad them to the same length
    X = pad_sequence([torch.tensor(v[0], dtype=torch.float32) for v in batch], padding_value=0)
    # store the y values and the length of each title as tensors
    y = torch.tensor([v[1] for v in batch], dtype=torch.float32)
    lens = torch.tensor([v[2] for v in batch], dtype=torch.float32)
    #print("X: Shape:  ",X.shape);
    #print("lens:  ", lens);
    return X, y, lens





"""
Use the GRU model, also consider bi-directional GRU model 

(probably not consider LSTM model memory test shows that LSTM has 
shorter memory than GRU)

Apply epoches to train
"""

# Prototype for GRU Classifier
class GRUClassifier(torch.nn.Module):
    def __init__(self, emb_size, hidden_size, num_layers, num_class):
        super().__init__();

        ## Initialise the neural network we will use in the forward method

        # converts characters into vectors (initially random vectors before training)
        #self.emb = nn.Embedding(num_chartype, emb_size, padding_idx=0)

        # applies a small amount of dropout to word embeddings
        self.drop = nn.Dropout(p = 0.2)

        # single layer GRU
        self.gru = nn.GRU(emb_size, hidden_size, num_layers = 1, bidirectional = False, batch_first = True);

        # MLP for outputing class logits
        self.W = nn.Linear(hidden_size, num_class)
    
    def forward(self, x, lens):
        
        #print("X Shape for x:    ", x.shape);
        #print("lens length:   ", lens.shape);

        # returns the embedding for each character
        #xemb = self.emb(x)
        xemb = self.drop(x);

        # pack the sequence to make the gru more efficient
        # packing needs the length of each sequence
        xemb = pack_padded_sequence(xemb, lens, enforce_sorted=False)
        _, rnn_out = self.gru(xemb)

        # apply the MLP to the last hidden output of each sequence
        out = self.W(rnn_out[-1]);
        return out





# Setting up the GRU model
hidden_size = 64;
# According to StackOverflow Formula
num_classes = 4;
hidden_layer = 4;
learning_rate = 0.005;
batch_size = 16;
clf = GRUClassifier(300, hidden_size, hidden_layer, num_classes);

# Set up the optimizer
opt = optim.Adam(clf.parameters(), lr=learning_rate);

loss_avg = None;
c = 0;

last_loss = 0;

shouldStop = False;
allow = 0;
last_f1_score = -1;
last_accuracy = -1;


while (not shouldStop):
#for i in range(1):
    
    clf.train();

    # Train Model
    data = DBPediaData(X_train_tokens_list, Y_training_data);
    data_loader = DataLoader(data, batch_size = batch_size, shuffle = True, collate_fn = collate_dbpedia);
    
    current_loss = 0;
    
    
    
    
    
    for X, y, lens in data_loader:
        
        
        #lens2 = lens.to("cpu");
        
        
        # run the GRU model forward
        out = clf(X, lens);
        
        # print("out: ", out)
        
        #out = out.to(device = device);
        
        # Return the result by using the argmax function
        y = y.type(torch.LongTensor)
    
        # compute the loss
        loss = nn.functional.cross_entropy(out, y, weight = class_weights);
    
        # zero gradients from last backward step
        opt.zero_grad()
    
        # compute gradients of the loss wrt the parameters (using backprop)
        loss.backward()
    
        # update the model parameters using Adam
        opt.step()
    
        # compute a smoothed loss term for printing
        if loss_avg is None:
            loss_avg = loss.item()
        else:
            loss_avg = loss_avg * 0.95 + loss.item() * 0.05
        c+=1
        if c % 500 == 0:
            print("Smooth loss:", loss_avg)
        
        current_loss += loss.item();
        
        
    
    
    ## Calculate the F1 macro and F1 micro
    clf.eval();
    
    # batch up the validation data
    valid_data = DBPediaData(X_validation_tokens_list, Y_validation_data);
    valid_data_loader = DataLoader(valid_data, batch_size=1, shuffle=False, collate_fn=collate_dbpedia)
    
    
    with torch.no_grad(): # turn of autograd to make everything faster

        all_chosen = [] # predictions made by the model
        all_y = [] # true classes (ground truth labels)
        for X, y, lens in valid_data_loader:
            
            #X = X.to(device=device);
            
            #y = y.to(device=device);
            
            # run the model forward
            out = clf(X, lens)

            # select the class with the highest score
            chosen = torch.argmax(out, dim=1);
            #chosen = chosen.to("cpu");
            #y = y.to("cpu");

            # store the predictions and ground truth labels
            all_chosen.append(np.array(chosen.numpy(), dtype=np.int32))
            all_y.append(np.array(y.numpy(), dtype=np.int32))
        
        # combine predictions, and true labels into two big vectors
        all_y = np.concatenate(all_y)
        all_chosen = np.concatenate(all_chosen)

        all_classes = [0,1,2,3];
        # plot the confusion matrix
        plt.figure(figsize=(12,8))
        ConfusionMatrixDisplay.from_predictions(all_y, all_chosen, display_labels=all_classes, ax=plt.gca())
        plt.xticks(rotation=45, ha="right")
        
        f1_macro = f1_score(all_y, all_chosen, average='macro');
        accuracy = f1_score(all_y, all_chosen, average='micro');

        # print the f1 scores
        print("F1 macro:", f1_macro);
        print("F1 micro:", accuracy);
        print("last_f1_score : ", last_f1_score);
        print("last accuracy: ", last_accuracy);
    
    current_loss = current_loss/len(X_training_data);
    print("current loss: ",current_loss);
    
    #if (last_f1_score > f1_macro and last_accuracy > accuracy):
    if (last_f1_score > f1_macro and current_loss < last_loss):
        allow += 1.5;
        last_f1_score = f1_macro;
        last_loss = current_loss;
    else:
        allow = max(0, allow-1);
        last_f1_score = f1_macro;
        last_accuracy = accuracy;
        last_loss = current_loss;
    
    
    # Testing shouldStop Condition
    if (allow>2):
        shouldStop = True;
    



# Fine Tune with Validation Data
clf.train();

# Train Model
validation_data = DBPediaData(X_validation_tokens_list, Y_validation_data);
validation_data_loader = DataLoader(validation_data, batch_size = 1, shuffle = True, collate_fn = collate_dbpedia);





for X, y, lens in validation_data_loader:
    
    #X = X.to(device=device);
    
    #y = y.to(device=device);
    
    # run the GRU model forward
    out = clf(X, lens)
    
    # Return the result by using the argmax function
    y = y.type(torch.LongTensor)

    # compute the loss
    loss = nn.functional.cross_entropy(out, y, weight=class_weights)

    # zero gradients from last backward step
    opt.zero_grad()

    # compute gradients of the loss wrt the parameters (using backprop)
    loss.backward()

    # update the model parameters using Adam
    opt.step()


clf.eval();




# predict on the test data
X_test_tokens_list = [];
y_test_label_fake = [];
test_data_index = 0;
while test_data_index < len(Xt):
    tokens_list = tokenizer.tokenize(Xt[test_data_index]);
    data_token_list = [];
    for token in tokens_list:
        if token in string.punctuation or token in stopwords:
            continue;
        else:
            data_token_list.append(token);
    if(len(data_token_list)==0):
        print("test data index: ", test_data_index);
        print("X:  ",Xt[test_data_index]);
        print("X data_token_list :  ", data_token_list);
    X_test_tokens_list.append(tokens_list);
    y_test_label_fake.append(1);
    test_data_index += 1;
Y_test_pred = [];
clf.eval();
test_data = DBPediaData(X_test_tokens_list, y_test_label_fake);
test_data_loader = DataLoader(test_data, batch_size=1, shuffle=False, collate_fn=collate_dbpedia);
with torch.no_grad():
    for X, y, lens in test_data_loader:
        ### Need to change
        if (lens!=0):
            output = clf(X,lens);
            chosen = torch.argmax(output, dim=1);
            #print(chosen);
            #print(chosen.item());
            #print(type(chosen.item()));
            Y_test_pred.append(chosen.item());
        else:
            chosen = 1;
            Y_test_pred.append(chosen);
        """
        print(chosen);
        print(chosen.item());
        print(type(chosen.item()));
        Y_test_pred.append(chosen.item());
        """
    print(len(Y_test_pred));
        




# write out the csv file
# first column is the id, it is the index to the list of test examples
# second column is the prediction as an integer
fout = open("out.csv", "w")
fout.write("Id,Predicted\n")
for i, line in enumerate(Y_test_pred): # Y_test_pred is in the same order as the test data
    fout.write("%d,%d\n" % (i, line))
fout.close()

