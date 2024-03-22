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
from transformers import DistilBertTokenizer, DistilBertModel;
from transformers import AutoModel;
from transformers import AutoModelForSequenceClassification;
from transformers import TrainingArguments, Trainer;


#Loading word2Vec Model
"""
Remember to Remove Limit Before Training The Whole Model

Also can try to train a word2vec model for not included words (probably not necessary).
Spend time on hyperparameters tuning and transformers.

path = 'C:/Users/user/Documents/Gensim Library/GoogleNews-vectors-negative300.bin.gz';
model = KeyedVectors.load_word2vec_format(path, binary=True);
word_included = model.key_to_index;
"""



#Initializing TreebankWordTokenizer
model_name = "distilbert-base-uncased";
tokenizer = DistilBertTokenizer.from_pretrained(model_name);
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu');
# device = 'cpu';

transformer_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels = 4);
# transformer_model = AutoModel.from_pretrained(model_name)

# load the training data
train_data = json.load(open("genre_train.json", "r"))
X = train_data['X']
Y = train_data['Y'] # id mapping: 0 - Horror, 1 - Science Fiction, 2 - Humour, 3 - Crime Fiction
docid = train_data['docid'] # these are the ids of the books which each training example came from

#X = X.to(device);

# load the test data
# the test data does not have labels, our model needs to generate these
test_data = json.load(open("genre_test.json", "r"))
Xt = test_data['X']
#print(Xt[0]);


# Calculating class weights
count_frequency = {};
for i in Y:
    if (i in count_frequency):
        count_frequency[i] += 1;
    else:
        count_frequency[i] = 1;
#print(count_frequency);
#print(len(Y));
keys = sorted(count_frequency.keys());
class_weights = [];
for key in keys:
    class_weights.append(1-(count_frequency[key]/len(Y)));

class_weights = torch.tensor(class_weights);
#print(class_weights);





# print(tokenizer(X_training_data[0], max_length=10, truncation=True));



###################################################
## Creating Text Embeddings using DistilBertModel##
###################################################

"""
training_x_data = [];
test_x_data = [];
with torch.no_grad():
    for text in X:
        #text = text.to(device);
        encoded_input = tokenizer(text, padding = True, truncation=True, return_tensors='pt', max_length=10);
        output = transformer_model(**encoded_input);
        last_hidden_state = output.last_hidden_state;
        training_x_data.append(last_hidden_state);
        print(last_hidden_state.shape);
torch.save(training_x_data, 'training.pt');
"""


#training_x_data = torch.load('training.pt');
def padding_for_hidden_layer(tensor_list):
    modify_training_x = [];
    for i in range(len(tensor_list)):
        zeros = torch.zeros(10,768);
        mat = torch.squeeze(tensor_list[i],0);
        zeros[:len(mat),:] = mat;
        modify_training_x.append(zeros);
    return modify_training_x;



"""
with torch.no_grad():
    for text in Xt:
        encoded_input = tokenizer(text, padding = True, truncation = True, return_tensors = 'pt');
        output = transformer_model(**encoded_input);
        last_hidden_state = output.last_hidden_state;
        test_x_data.append(last_hidden_state);
"""



#modify_training_x=padding_for_hidden_layer(training_x_data);


"""
for i in range(1000):
    print(modify_training_x[i].shape);
#print(len(test_x_data));
"""



# Spliting training data into training and validation data
train_val_number = int(len(X)*0.7);
#X_training_data = modify_training_x[:train_val_number];
X_training_data = X[:train_val_number];
#X_validation_data = modify_training_x[train_val_number:];
X_validation_data = X[train_val_number:];
Y_training_data = Y[:train_val_number];
Y_validation_data = Y[train_val_number:];


# Turning raw data into tokenized
"""
train_encodings = [];
for i in X_training_data:
    a = tokenizer(i,truncation = True, padding = True);
    train_encodings.append(a);
validation_encodings = [];
for j in X_validation_data:
    b = tokenizer(j, truncation = True, padding = True);
    validation_encodings.append(b);
train_encodings = np.array(train_encodings);
validation_encodings = np.array(validation_encodings);
"""
train_encodings = tokenizer(X_training_data, truncation = True, padding = True);
validation_encodings = tokenizer(X_validation_data, truncation = True, padding = True);






"""
encoded_input = tokenizer(X[0], return_tensors= 'pt');
hidden_state = transformer_model(**encoded_input);
last_hidden_state = hidden_state.last_hidden_state;
print("hidden state: ", type(last_hidden_state));
print(last_hidden_state.shape);
"""






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


# Converting sequence into encodings that can be fed into pre-trained transformer
def convert_sequence_to_input(sequence):
    sequence_encoding = tokenizer(sequence);
    input_IDs = sequence_encoding['input_ids'];
    attention_mask_list = sequence_encoding['attention_mask'];
    return input_IDs, attention_mask_list;

    



class DataSet(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        # the titles in the data (our unprocessed X values)
        self.encodings = encodings; 
        # the entity types in the data (our y)
        self.labels = labels;
    
    def __len__(self):
        return len(self.encodings);
    
    def __getitem__(self, index):
        item = {key: torch.tensor(val[index]) for key, val in self.encodings.items()};
        item['labels'] = torch.tensor(self.labels[index]);
        return item;
        """
        X = torch.flatten(self.input[index]);
        sum_torch = torch.Tensor(X[0].shape);
        for i in X:
            sum_torch+=i;
        #Note: we also return the length of the title to help with padding
        return sum_torch, self.labels[index];
        """


def compute_method(eval_output):
    logits, y_true = eval_output;
    print("logits: ", logits);
    print("y_true: ", y_true);
    predictions_of_model = [];
    for i in logits:
        prediction = np.argmax(i);
        predictions_of_model.append(prediction);
    print("prediction of model : ",predictions_of_model);
    score = f1_score(y_true, predictions_of_model, average="macro");
    return {'f1 score' : score};



## Creating the training and evaluation dataset
# print(train_encodings)
training_dataset_torch = DataSet(train_encodings, Y_training_data);
validation_dataset_torch = DataSet(validation_encodings, Y_validation_data);


training_arguments = TrainingArguments(evaluation_strategy="epoch", 
                                       output_dir = "transformer_output",
                                       do_train = True,
                                       eval_accumulation_steps=0);


training_funct = Trainer(
    model = transformer_model,
    args = training_arguments,
    train_dataset = training_dataset_torch,
    eval_dataset = validation_dataset_torch,
    compute_metrics = compute_method
    )

training_funct.train();

    

# Not useful since padding is already done in tokenization part

"""
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




## Neural Network does not appear to perform well

"""
## Creating a simple neural network to map transformer output to classifier
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NeuralNetwork, self).__init__();
        #self.dropout = nn.Dropout(0.2);
        self.layer1 = nn.Linear(input_size, 5000);
        self.layer2 = nn.Linear(5000, 3000);
        #self.layer3 = nn.Linear(12000, 3000);
        self.layer4 = nn.Linear(3000, 400);
        self.layer5 = nn.Linear(400, 100);
        self.layer6 = nn.Linear(100, 10);
        self.layer7 = nn.Linear(10, num_classes);
    def forward(self, x):
        #x = self.dropout(x);
        x = nn.functional.relu(self.layer1(x));
        x = nn.functional.relu(self.layer2(x));
        # x = nn.functional.relu(self.layer3(x));
        x = nn.functional.relu(self.layer4(x));
        x = nn.functional.relu(self.layer5(x));
        x = nn.functional.relu(self.layer6(x));
        x = self.layer7(x);
        return x;
"""





"""
# Setting up the GRU model
hidden_size = 64;
# According to StackOverflow Formula
num_classes = 4;
hidden_layer = 8;
learning_rate = 0.005;
batch_size = 16;
number_of_features = 10*768
# clf = GRUClassifier(300, hidden_size, hidden_layer, num_classes);

neural_network = NeuralNetwork(number_of_features, num_classes).to(device);

# Set up the optimizer
# opt = optim.Adam(clf.parameters(), lr=learning_rate);

# Setting loss and optimizer
criterion = nn.CrossEntropyLoss(weight = class_weights);
optimizer = optim.Adam(neural_network.parameters(), lr = learning_rate);


loss_avg = None;
c = 0;

last_loss = 0;

shouldStop = False;
allow = 0;
last_f1_score = -1;
last_accuracy = -1;


while (not shouldStop):
#for i in range(1):
    
    neural_network.train();
    
    #clf.train();

    # Train Model
    # data = DBPediaData(X_train_tokens_list, Y_training_data);
    # data_loader = DataLoader(data, batch_size = batch_size, shuffle = True, collate_fn = collate_dbpedia);
    
    data = DataSet(X_training_data, Y_training_data);
    data_loader = DataLoader(data, batch_size = batch_size, shuffle = True);
    
    current_loss = 0;
    
    
    
    
    
    for X, y in data_loader:
        
        X = X.to(device=device);
        
        y = y.to(device=device);
        
        # print(X.shape);
        
        
        # Forward
        output = neural_network(X);
        loss = criterion(output, y);
        
        # backward
        optimizer.zero_grad();
        loss.backward();
        
        # gradient descent or adam step
        optimizer.step();
        
        
        # run the GRU model forward
        # out = clf(X, lens)
        
        # Return the result by using the argmax function
        #y = y.type(torch.LongTensor)
        
        """
        
        
        
        
        # Not sure what is this for
    
"""
        # compute the loss
        loss = nn.functional.cross_entropy(out, y)
    
        # zero gradients from last backward step
        opt.zero_grad()
    
        # compute gradients of the loss wrt the parameters (using backprop)
        loss.backward()
    
        # update the model parameters using Adam
        opt.step()
        """
    
    
"""
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
    #clf.eval();
    neural_network.eval();
    
    # batch up the validation data
    valid_data = DataSet(X_validation_data, Y_validation_data);
    valid_data_loader = DataLoader(valid_data, batch_size=1, shuffle=False);
    
    
    with torch.no_grad(): # turn of autograd to make everything faster

        all_chosen = [] # predictions made by the model
        all_y = [] # true classes (ground truth labels)
        for X, y in valid_data_loader:
            
            X = X.to(device = device);
            
            y = y.to(device = device);
            
            # run the model forward
            output = neural_network(X);

            # select the class with the highest score
            chosen = torch.argmax(output, dim=1);
            chosen = chosen.to("cpu");
            y = y.to("cpu");

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

    """




"""

# Fine Tune with Validation Data
clf.train();

# Train Model
validation_data = DBPediaData(X_validation_tokens_list, Y_validation_data);
validation_data_loader = DataLoader(validation_data, batch_size = 1, shuffle = True, collate_fn = collate_dbpedia);





for X, y, lens in validation_data_loader:
    
    # run the GRU model forward
    out = clf(X, lens)
    
    # Return the result by using the argmax function
    y = y.type(torch.LongTensor)

    # compute the loss
    loss = nn.functional.cross_entropy(out, y)

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
        if token in string.punctuation:
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
        
        # print(chosen);
        # print(chosen.item());
        # print(type(chosen.item()));
        # Y_test_pred.append(chosen.item());
        
    print(len(Y_test_pred));
    
    
"""
        


"""

# write out the csv file
# first column is the id, it is the index to the list of test examples
# second column is the prediction as an integer
fout = open("out.csv", "w")
fout.write("Id,Predicted\n")
for i, line in enumerate(Y_test_pred): # Y_test_pred is in the same order as the test data
    fout.write("%d,%d\n" % (i, line))
fout.close()

"""

