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

In this project, I analysed and trained twitter airline sentiment dataset which contains roughly 1.5k rows. 
There are 3 sentiments(negative,neutral,postiive) classes and implimented both multinomianl and binally logistic regression to classify this.
In order to apply the machine learning algorithm, tokenized text data and then applied Tf-idf to transofrom the tokens into numbers in matrices.

After the training, analized results and hypothetized potenital solutions to the weakness of Tf-idf which found through analysis.


## Motivation

- Explore the text dataset 
- Apply data science to real word data
- Implement Tf-idf from scratch(no sklearn) 
- Compare the performance of multinomial logtistic regression with binally logistic regression


## File_Introduction

- [Sentiment_Analysis.ipynb](https://github.com/shchuka555/sentiment-analysis/blob/main/Sentiment_Analysis.ipynb): Data Analysis done using the kaggle dataset<br>
- [Tweets.csv](https://github.com/shchuka555/sentiment-analysis/blob/main/Tweets.csv): Tweeter-arline-sentiment dataset which obtained in February 2015 through skrapiTechnical Aspect


## Technical_Aspect

- ### Step1: Use top 2000 frequently appeard words in training dataset to make Tf-idf

    Since there are roughly unique 15k words and punctuations in the training dataset, the 
    computational cost of the traning will be larger if we make Tf-idf for all unique words. 
    Thus, decided to use top 2000 frequent word.
   
- ### Step2: Stopwords: removing some of the words from counting
  
  ntlk library has own sets of stopwords but choose not to use it because the sentiment analysis about tweets.
  Unlike to natural languahe processing analysis like classifying category of news articles, the classification process does not 
  depend on specific terminologies. Rather, we want to keep ordinary expression like "good" and "bad".
  
  However, still tt will be better to remove some words and punctuations which has almsot 0 correlation with sentiment of tweets.
  I created my own set of stopwrods which are 
    - [;],[I], [.], [,],[&],[:],[#],[...],['']
    - [@] and name of airlines which contain in all tweets.
  
  **Reasons**: 
  Keeping the airline names will result in lower classification accuracy.
  The plots in [Dataset_Analysis](#Dataset_Analysis) show majority of tweets in the dataset are classified as negative.
  Therefore, the algorithm will learn them as negative words despite airline names have almost 0 correlation with sentiment of the tweet.
  
  
  **Non Stopwords**: 
  On the other hand I kept some punctuations which often used with strong emotion.
  For example [?] and [!] 
   
  code:
 
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
   

- ### Step3: Appying Tf-idf
  
  I implemented standard Tf and smooth idf without relying on sklearn.
    - smooth idf prevents division of 0 even if a word in a testing dataset does not appear in top 2000 frequent word.
  
  **code:••
  
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
  
  Applied logstic regresson because I want to compare performace of the multionminal and bianlly classificaiton.
  Set max iteration as 500 to let algorithm finds coefficients such that maximize classification accuracy.
  
  **code:••
  
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


This plot tells few thiings about sentiment and arlines.

- negative review is dominant sentiment among all airlines in the dataset.
- United, US airline, American have many bad reviews relative to other airlines.

## Multinominal_Logistic_Regression


![https://github.com/shchuka555/sentiment-analysis/blob/main/samples/output.png](https://github.com/shchuka555/sentiment-analysis/blob/main/samples/output.png)

- Multinominal logistic regression correctly classified **92.5%** of negative tweets. 
- This algorithm poorly perform to classify Neutral and Positive classes.
  - **25.4%** of neural tweets classifed as negaitve 
  - **12.8%** of positive tweets classifed as negative

## Binally_Logistic_Regression

![https://github.com/shchuka555/sentiment-analysis/blob/main/samples/newplot-5.png](https://github.com/shchuka555/sentiment-analysis/blob/main/samples/newplot-5.png)

- Binally logistic regression improved overall accuracy compared wtih multinominal case 
- There are rouhgly  **10%** of improvement in both training and testing accuracy


![https://github.com/shchuka555/sentiment-analysis/blob/main/samples/output2.png](https://github.com/shchuka555/sentiment-analysis/blob/main/samples/output2.png)

- correctly classified **97.7%** of negative tweets. 
- However, still **14.3%** of positive tweets classifed as negative


## Analysis_of_Regression_Coefficients

![https://github.com/shchuka555/sentiment-analysis/blob/main/samples/newplot-6.png](https://github.com/shchuka555/sentiment-analysis/blob/main/samples/newplot-6.png)

![https://github.com/shchuka555/sentiment-analysis/blob/main/samples/newplot-7.png](https://github.com/shchuka555/sentiment-analysis/blob/main/samples/newplot-7.png)

![https://github.com/shchuka555/sentiment-analysis/blob/main/samples/newplot-8.png](https://github.com/shchuka555/sentiment-analysis/blob/main/samples/newplot-8.png)




## Analysis_of_Misclassified_Tweets

  - ### Positive tweet which predicted as negative
    
    #### '@ united all flights cancelled flighted: (trip refunded without difficulty, staff extremely helpful, no complaints! way to handle bad weather!'

    Obtained above tweet by sorting postive tweet by predicted probability that belonges to negative class.
    (the tweet predicted as 99.8% negative tweet)
    The tweet is clearly postive because the user complemented how it was smooth to get refund from airline and staff was helpful.
    However the prediction is negative, the following plot explans the reason why it predicted as negative.

    ![https://github.com/shchuka555/sentiment-analysis/blob/main/samples/newplot-9.png](https://github.com/shchuka555/sentiment-analysis/blob/main/samples/newplot-9.png)

    The plot shows words like "Without" and "Complained" have really negative coefficients. 
    Despite there are many postive words in the tweet, the magitude is too small compared with negative words. 
    Therefore such words pulled the prediction toward negative direction and the prediction is extreme.

  - ### Negative tweet which predicted as positive

    #### '@ americanair thanks for info on super large passengers- the extra seat mr. big needed was the one i was sitting in already #customerservice'
  
    Obtained above tweet by sorting negative tweet by predicted probability that belonges to positive class.
    (This is not "most misclassifed tweet" but the tweet predicted as 99.4% positive tweet)
    
    The above tweet is negative but the sarcastic consisted by many posotive words.
    
    Therefore, algorithm predicted the tweet as positive, the following plot validate the statement

    ![https://github.com/shchuka555/sentiment-analysis/blob/main/samples/newplot-10.png](https://github.com/shchuka555/sentiment-analysis/blob/main/samples/newplot-10.png)



## Conclusion

  - ## Results
    -  binally logistic regression have lower error rate than multinomial one.
    -  The algorithm has lower accuracy on classifying non negative tweets.

  - ## Issues
    - positve phrase can be viewed as 2 negative words 
      - example: "without complained" -> "without" + "complained"
    - the approach with Tf-idf can misclassify sarcastic negative tweet which contains many postive words.
 
  - ## potential solutions
    - applying bi-gram Tf-Idf instead of uni-gram
    - the issue with sarcastic text can be hard to deal with but it might not exist in Non-English tweets sentiment analysis
      - for example, sarcasm is far less common in Japanese language and culture.

## Credits

Dataset from [Kaggle](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment):
Udeny course [natural-language-processing-in-python](https://www.udemy.com/course/natural-language-processing-in-python/):
