# Sentiment Analysis with Logistic Regression

## Table of Contents

- [Overview](#Overview)  
- [Motivation](#Motivation)
- [File_Introduction](#File_Introduction)  
- [Technical_Aspect](#Technical_Aspect)  
- [Analysis](#Analysis)
  - [Dataset_Analysis](#Dataset_Analysis)
  - [Multinominal_Logistic_Regression](#Multinominal_Logistic_Regression)
  - [Binally_Logistic_Regression](#Binally_Logistic_Regression)
  - [Analysis_of_Regression_Coefficients](#Analysis_of_Regression_Coefficients)
  - [Analysis_of_Misclassified_Tweets](#Analysis_of_Misclassified_Tweets)
- [Conclusion](#Conclusion)  
- [Credits](#Credits)  




## Overview 

In this project, I analyzed and trained the Twitter airline sentiment dataset, which contains roughly 1.5k rows. 
There are three sentiments (negative, neutral, and positive) classes and implemented both multinomial and binary logistic regression to classify sentiments. To apply the machine learning algorithm, tokenized text data and then used Tf-IDF to transform the tokens into numbers in matrices.

After the training, analyzed results and hypothesized potential solutions to the weakness of Tf-IDF which was found through analysis.

## Motivation

- Explore the text dataset.
- Apply data science to real world data.
- Implement Tf-IDF from scratch (no sklearn).
- Compare the performance of multinomial logistic regression with binary logistic regression.


## File_Introduction

- [Sentiment_Analysis.ipynb](https://github.com/shchuka555/sentiment-analysis/blob/main/Sentiment_Analysis.ipynb): Data Analysis done using the kaggle dataset<br>
- [Tweets.csv](https://github.com/shchuka555/sentiment-analysis/blob/main/Tweets.csv): kaggle Tweeter-arline-sentiment dataset which obtained in February 2015

## Technical_Aspect

- ### Step1: Use the top 2000 frequently appeared words in training dataset to make Tf-idf.

    Since there are roughly unique 15k words and punctuations in the training dataset, the 
    computational cost of the training will be larger if we make Tf-IDF for all unique words. 
    Thus, decided to use the top 2000 frequent words.
   
- ### Step2: Stopwords: removing some of the words from counting
  
  ntlk library has its own set of stopwords, but I choose not to use it because of the sentiment analysis about tweets.
  Unlike natural language processing analysis, like classifying the category of news articles, the classification process does not 
  depend on specific terminologies. Instead, we want to keep ordinary expressions like "good" and "bad."
  
  However, it will still be better to remove some words and punctuations that have almost 0 correlation with the sentiment of tweets.
  I created my own set of stopwords which are 
    - [;],[I], [.], [,],[&],[:],[#],[...],['']
    - [@] and names of airlines contains in all tweets.
  
  **Reasons**: 
  Keeping the airline names will result in lower classification accuracy.
  The plots in [Dataset_Analysis](#Dataset_Analysis) show majority of tweets in the dataset are classified as negative.
  Therefore, the algorithm will learn them as negative words despite airline names having almost 0 correlation with the tweet's sentiment.
  
  
  **Non Stopwords**: 
  On the other hand, I kept some punctuation often used with strong emotion.
  For example [?] and [!].
   
  **code:**
 
  ```python 
  ### function which returns top 2000 frequent words and it's counts in the train set.
  def get_top2000(X):
    airline_name = list(map(lambda company: company.lower().replace(' ',''), ['airline']))
    airline_name.append('@')
    ### Lists of "words" which are too common and preferred to be removed.
    remove = {'.','I',',',';','&',':','#','...',"''",}
    airline_name = list(set(airline_name))
    counter = dict()
    for text in X:
        for word in text:
            if word in chain(airline_name,remove):
                None
            elif word not in counter:
                counter[word] = 1   
            else: 
                counter[word] += 1
    top_2000 = sorted(counter,key=counter.get)[-2000:]
    return(top_2000,counter)
  
  top_2000,counter = get_top2000(x_train)
  ```
   

- ### Step3: Appying Tf-IDF
- 
  I implemented standard Tf and smooth IDF without relying on sklearn.
    - smooth IDF prevents the division of 0 when a word in a testing dataset does not appear in the top 2000 frequent words.
 
  **code:**
  
  ```python
    #### self made Tf_idf with smooth idf 
    def Tf_idf(X,top_2000,counter):
      Tf = np.zeros((len(X),len(top_2000)))
      for i,tweet in enumerate(X):
        for word in tweet:
          if word in top_2000:
            Tf[i,top_2000.index(word)] = counter[word] 
            
      n_t = np.zeros((len(X),len(top_2000)))
      for i,tweet in enumerate(X):
        for word in top_2000:
          if word in tweet:
            n_t[i,top_2000.index(word)] += 1
            
      ## adding 1 to prevent division by zero
      smooth_idf = np.log(len(X) / (np.sum(n_t,axis=0,keepdims=True)+1)) + 1
      tf_idf = Tf*smooth_idf
      return (tf_idf)
  
  ```  
  
  
- ### Step4: Implement Logsitic regression
  
  Applied logistic regression because I wanted to compare the multinomial and binary classification performance.
  Set max iteration as 500 to let the algorithm find coefficients that maximize classification accuracy.
  
  **code:**
  
  ```python
  ### apply LogisticRegression with 500 iteration
  from sklearn.linear_model import LogisticRegression
  model = LogisticRegression(max_iter=500)
  model.fit(tf_idf_train,y_train)
  print('train accuracy:',model.score(tf_idf_train,y_train))
  print('test  accuracy:',model.score(tf_idf_test,y_test))
  ```
  
  
## Analysis

## Dataset_Analysis


![https://github.com/shchuka555/sentiment-analysis/blob/main/samples/newplot-2.png](https://github.com/shchuka555/sentiment-analysis/blob/main/samples/newplot-2.png)

![https://github.com/shchuka555/sentiment-analysis/blob/main/samples/newplot-3.png](https://github.com/shchuka555/sentiment-analysis/blob/main/samples/newplot-3.png)

To learn more about the dataset, I grouped the sentiments by airlines.
![https://github.com/shchuka555/sentiment-analysis/blob/main/samples/newplot-4.png](https://github.com/shchuka555/sentiment-analysis/blob/main/samples/newplot-4.png)


This plot tells a few things about sentiment and airlines.

- Negative review is a dominant sentiment among all airlines in the dataset.
- United, US Airlines and American have many bad reviews relative to other airlines.

## Multinomial_Logistic_Regression


![https://github.com/shchuka555/sentiment-analysis/blob/main/samples/output.png](https://github.com/shchuka555/sentiment-analysis/blob/main/samples/output.png)

- Multinomial logistic regression correctly classified **92.5%** of negative tweets.
- This algorithm poorly performs in classifying neutral and positive classes.
  - **25.4%** of neural tweets classified as negative.
  - **12.8%** of positive tweets classified as negative.

## Binary_Logistic_Regression

![https://github.com/shchuka555/sentiment-analysis/blob/main/samples/newplot-5.png](https://github.com/shchuka555/sentiment-analysis/blob/main/samples/newplot-5.png)

- Binary logistic regression improved overall accuracy compared with the multinomial case. 
- There are roughly **10%** of improvements in both training and testing accuracy.


![https://github.com/shchuka555/sentiment-analysis/blob/main/samples/plot.png](https://github.com/shchuka555/sentiment-analysis/blob/main/samples/plot.png)

- Correctly classified **97.7%** of negative tweets.
- However, still **14.3%** of positive tweets classified as negative.


## Analysis_of_Regression_Coefficients

![https://github.com/shchuka555/sentiment-analysis/blob/main/samples/newplot-6.png](https://github.com/shchuka555/sentiment-analysis/blob/main/samples/newplot-6.png)

![https://github.com/shchuka555/sentiment-analysis/blob/main/samples/newplot-7.png](https://github.com/shchuka555/sentiment-analysis/blob/main/samples/newplot-7.png)

![https://github.com/shchuka555/sentiment-analysis/blob/main/samples/newplot-8.png](https://github.com/shchuka555/sentiment-analysis/blob/main/samples/newplot-8.png)



## Analysis_of_Misclassified_Tweets

  - ### A positive tweet which predicted to negative
    
    #### '@ united all flights cancelled flighted: (trip refunded without difficulty, staff extremely helpful, no complaints! way to handle bad weather!'

    Obtained the above tweet by sorting positive tweets by the predicted probability belonging to the negative class.
    (The binary regression prediction is 99.8% negative)
    The tweet is clearly positive because the user complemented how it was smooth to get a refund from the airline, and the staff was helpful.
    However, the prediction is negative. The following plot explains the reason why it was predicted as negative.

    ![https://github.com/shchuka555/sentiment-analysis/blob/main/samples/newplot-9.png](https://github.com/shchuka555/sentiment-analysis/blob/main/samples/newplot-9.png)

    The plot shows words like "Without" and "Complained" have really negative coefficients. 
    Despite many positive words in the tweet, the magnitude is too small compared to negative words. 
        Therefore such words pulled the prediction in a negative direction.

  - ### A negative tweet which predicted to positive

    #### '@ americanair thanks for info on super large passengers- the extra seat mr. big needed was the one i was sitting in already #customerservice'
  
    Obtained the above tweet by sorting negative tweets by the predicted probability that belongs to the positive class.
    (This is not the "most misclassified tweet," but the algorithm predicted to 99.4% positive tweet)
    The above tweet is negative, but the sarcasm consists of many positive words.
    Therefore, the algorithm predicted the tweet as positive.

    ![https://github.com/shchuka555/sentiment-analysis/blob/main/samples/newplot-10.png](https://github.com/shchuka555/sentiment-analysis/blob/main/samples/newplot-10.png)

  

## Conclusion

  - ## Results
    -  Binary logistic regression has a lower error rate than multinomial one.
    -  Both algorithms have lower accuracy in classifying non-negative tweets.
    - The unigram model approach treats some positive phrases as multiple negative words.
      - example: "without complained" -> "without" + "complained"
    - The approach with Tf-IDF misclassified sarcastic and negative tweet, which contains many positive words.
 
  - ## Potential solutions
    - Applying bigram Tf-IDF instead of uni-gram.
    - The issue with sarcastic text can be hard to deal with, but it might not exist in Non-English tweets sentiment analysis.
      - for example, sarcasm is far less common in Japanese language and culture.

## Credits

Dataset from [Kaggle](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment):
Udeny course [natural-language-processing-in-python](https://www.udemy.com/course/natural-language-processing-in-python/):

