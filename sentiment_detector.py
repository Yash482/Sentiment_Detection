#getting the dataset
g = open('reviews.txt' , 'r') # what we know
reviews = list(map(lambda x: x[:-1], g.readlines()))
g.close()

g = open('labels.txt' , 'r') # what we want to know
labels = list(map(lambda x: x[:-1].upper(), g.readlines()))
g.close()

"""
print(reviews[1])
print(labels[1])
"""

from collections import Counter
from concept import pos_neg_words
import numpy as np
import time
import sys

total_counts = Counter()
pos_neg_ratios = Counter()

total_counts, pos_neg_ratios = pos_neg_words(reviews, labels)

#Building NN

class NeuralNetwork :
    def __init__(self, reviews, labels, hidden_nodes, min_count =10, polarity_cutoff = 0.1,  learning_rate = 0.1):
        np.random.seed(1)
        self.pre_process_data(reviews, polarity_cutoff, min_count)
        self.init_network(len(self.reviews_vocab), hidden_nodes, 1, learning_rate)
        
    def pre_process_data(self, reviews, polarity_cutoff, min_count):
        #get review vocab
        reviews_vocab = set()
        for review in reviews :
            for word in review.split(" "):
                if total_counts[word] > min_count:
                    if word in pos_neg_ratios.keys():
                        if ((pos_neg_ratios[word] >= polarity_cutoff) or (pos_neg_ratios[word] <= -polarity_cutoff)):
                            reviews_vocab.add(word)
                    else:
                        reviews_vocab.add(word)
        self.reviews_vocab = list(reviews_vocab)
        #our review vocab is ready
        #it contains less noise
        #by polarity cutoff, we elimate those words with high frequency
        #becoz words like 'the' 'is' etc comes in large no of times comparatively
        
        #Same for label
        label_vocab = set()
        for label in labels:
            label_vocab.add(label)
        self.label_vocab = list(label_vocab)
        
        """
        Now the last job in data pre processing.
        As we get the data, now we have to assign each word to a no. to input in our NN
        """
        self.word2index = {} #it will be a dict
        for i, word in enumerate(self.reviews_vocab):
            self.word2index[word] = i
            
        self.label2index = {} #it will be a dict
        for i, label in enumerate(self.label_vocab):
            self.label2index[word] = i
    
    #data pre processing is done
    #Now we have to initialize our network
    
    def init_network(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.learning_rate = learning_rate
        
        #initializing weights
        self.weights_0_1 = np.zeros((self.input_nodes, self.hidden_nodes))
        self.weights_1_2 = np.random.normal(0.0, self.output_nodes**-0.5,(self.hidden_nodes, self.output_nodes))
        
        #initialize input layer
        self.layer_0 = np.zeros((1, input_nodes))
        self.layer_1 = np.zeros((1, hidden_nodes))
        #our input layer consist of 1 node for each word in our vocab
        #and which word corresponds to which node, to get this we have word2index
        
        
    #Our network is initialized
    #Now we define some functions which will be needed while training
    def update_input_layer(self, review):
        self.layer_0 *= 0
        for word in review.split(" "):
            if word in self.word2index.keys():
                self.layer_0[0][self.word2index[word]] = 1
         #this will be needed for each review while training
    
    #what should be our output for a label while training
    def get_target_for_label(self, label):
        if label == 'POSITIVE':
            return 1
        else:
            return 0
    
    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))
    
    def sigmoid_output_2_derivative(self, output):
        return output*(1-output)
    
    #Now everything is ready
    #2 things left: Training and Testing
    
    def train(self, training_reviews_raw, training_labels):
        training_reviews = list()
        for review in training_reviews_raw:
            indices = set()
            for word in review.split(" "):
                if word in self.word2index.keys():
                    indices.add(self.word2index[word])
            training_reviews.append(list(indices))
        #we did this to increase computation time
        #now we dont't have to loop thru every input node for making hidden layer
        #we could use this training reviews wich have indices of wrd present in each review
        
        #arise error if it doesn't satisfy
        assert(len(training_reviews) == len(training_labels))
        
        correct_so_far =0
        start = time.time()
        #Now for each of the training review we need to train our NN
        for i in range(len(training_reviews)):
            review = training_reviews[i]
            label = training_labels[i]
            
            #Implementing Forward Propagation
            
            #Input layer
            #we don't do anything here. as it is not req, we have indices to get layer1
            
            #Hidden Layer
            self.layer_1 *= 0
            for index in review:
                self.layer_1 += self.weights_0_1[index]
                
            #Output layer
            layer_2 = self.sigmoid(self.layer_1.dot(self.weights_1_2))
            
            #Implementing Backward Propagation
            
            #Output error
            layer_2_error = layer_2 - self.get_target_for_label(label)
            layer_2_delta = layer_2_error * self.sigmoid_output_2_derivative(layer_2)
            
            #Backpropagated Error
            layer_1_error = layer_2_delta.dot(self.weights_1_2.T)
            layer_1_delta = layer_1_error #no nonlinerity, so same
            
            #Update weights
            self.weights_1_2 -= self.layer_1.T.dot(layer_2_delta) * self.learning_rate
            for index in review:
                self.weights_0_1[index] -= layer_1_delta[0] * self.learning_rate 
            
            if np.abs(layer_2_error) < 0.5:
                correct_so_far += 1
            
            if float(time.time() - start)== 0:
                reviews_per_sec =0
            else:
                reviews_per_sec = i/float(time.time() - start)
            
            sys.stdout.write("\rProgress: " + str(100* i/float(len(training_reviews)))[:4] + "% Speed(reviews/sec): " + str(reviews_per_sec) + " Accuracy: " + str(correct_so_far/(i+1)))
            
            if(i%2500 ==0):
                print(" ")
               
            #Training is done++++++++++
                
    def test(self, testing_reviews, testing_labels):
        correct =0
        start = time.time()
        for i in range(len(testing_reviews)):
            pred = self.run(testing_reviews[i])
            if pred == testing_labels[i]:
                correct += 1
            
            if float(time.time() - start)== 0:
                reviews_per_sec =0
            else:
                reviews_per_sec = i/float(time.time() - start)
            
            
            sys.stdout.write("\rProgress: " + str(100* i/float(len(testing_reviews)))[:4] + "% Speed(reviews/sec): " + str(reviews_per_sec)[0:5] + "%Correct: " + str(correct) + "#Tested: " + str(i+1) +" Testing Accuracy: " + str(correct/(i+1)))                        

            
    def run(self, review):
        self.update_input_layer(review)
        self.layer_1 = self.layer_0.dot(self.weights_0_1)
        layer_2 = self.sigmoid(self.layer_1.dot(self.weights_1_2))
        if layer_2[0] > 0.5:
            return 'POSITIVE'
        else:
            return 'NEGATIVE'
        

nn = NeuralNetwork(reviews[:-1000], labels[:-1000], hidden_nodes= 10, min_count = 20, polarity_cutoff = 0.3, learning_rate = 0.1)
nn.train(reviews[:-1000], labels[:-1000])
nn.test(reviews[-1000:], labels[-1000:])
                    
        












