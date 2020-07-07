# Sentiment_Detection
A neural network which detects the sentiment of a movie review 
Here we take 2500 IMDB movie reviews and we build a deep neural network from scratch which predict the sentiment of each movie review.

We saw words like flawless, wonderful, beautiful etc. implies a POSITIVE review whereas words like trash, terrible etc. implies NEGATIVE one.
We first count the different words and made one node for each word in out input layer. Then for each review, then for each word in it, we do +1 for each word in it corresponding node.
After implementing everything, we get a accuracy of just 50% approx.

Accuracy was low due to noise.
Words like 'the' , 'is' , 'to' etc. comes more and in both kind of review. This affects the learning of neural network. 
Even if they are given less weight, their count is more( as we did +1 for each appearance in a review) and the do more negative.
To reduce it, we do =1 for each word. This gives equal importance to each word and accuracy increases to approx 80%. 

Further, we apply polarity cutoff which increases accuracy to 85%.
