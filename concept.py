

import numpy as np
from collections import Counter

def pos_neg_words(reviews, labels):
    positive_counts = Counter() 
    negative_counts = Counter()
    total_counts = Counter()

    #we do this to decrease the noise ie the unwanted words like 'the' 'is' etc

    #1st get the words that is in positive and negative reviews
    for i in range(len(reviews)):
        if labels[i]=='POSITIVE' :
            for word in reviews[i].split(" "):
                positive_counts[word] += 1
                total_counts[word] += 1
        else :
            for word in reviews[i].split(" "):
                negative_counts[word] += 1
                total_counts[word] += 1
    
    #Remove common words like 'the' 'is' etc        
    pos_neg_ratios = Counter()
    for term, cnt in list(total_counts.most_common()):
        if cnt >50:
            pos_neg_ratio = positive_counts[term]/float(negative_counts[term] +1)
            pos_neg_ratios[term] = pos_neg_ratio
        
    for wrd, ratio in list(pos_neg_ratios.most_common()):
        if ratio > 1: # => this word leads to possitive sentiment
            pos_neg_ratios[wrd] = np.log(ratio)
        else:
            pos_neg_ratios[wrd] = -np.log((1/(ratio + 0.01)))
            
    return (total_counts, pos_neg_ratios)

#run this to check our theory works
#print(list(pos_neg_ratios.most_common()))

"""
By doing this, we reduce a lot of noise by reducing vocab size
this will prevent our NN to work on wrong data
this increasescomputation time and efficiency od our NN
"""
