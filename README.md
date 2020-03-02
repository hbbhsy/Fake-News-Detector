# Fake-News-Detector

![title](img/fakenews.jpg)

Today we are living in the Era of information explosion. Along with the development of the Internet, the emergence and widespread adoption of the social media concept have changed the way news is formed and published. 

News has become faster, less costly and easily accessible with social media. This change has come along with some disadvantages as well. In particular, beguiling content, such as fake news made by social media users, is becoming increasingly dangerous. The fake news problem, despite being introduced for the first time very recently, has become an important research topic due to the high content of social media. Writing fake comments and news on social media is easy for users. The main challenge is to determine the difference between real and fake news. 

## Scope
One of the most significant limitations of this work is the data that is openly available. The focus of this work is on using novel deep learning based approaches for natural language processing. Some of them (like LSTMs based on word embeddings) usually require large quantities of data.

![info-explosion](img/Information-Explosion.jpg)

## Data

The [Fake News Corpus](https://github.com/several27/FakeNewsCorpus) is an open source dataset composed of millions of news articles mostly scraped from a curated list of 1001 domains from http://www.opensources.co/. 

The corpus was created by scraping (using scrapy) all the domains as provided by http://www.opensources.co/. Then all the pure HTML content was processed to extract the article text with some additional fields (listed below) using the newspaper library. Each article has been attributed the same label as the label associated with its domain.

Click [here](data/sample.csv) for a small representative shard of example data.

Stopwords used for text preprocess is downloaded from the useful repo [news-stopwords](https://github.com/vikasing/news-stopwords.git). Only [1000 stopwords for news](data/sw1k.csv) were used for this project.

## EDA
The news are labeled into 8 types. For each type, the 10 most frequent words are:

**Type**|**Words**
----|-----
rumor|wave, theory, wheeler, illusion, experiment, frequency, quantum, researcher, delayedchoice, physicist
hate|muslim, islam, palestinian, refugee, migrant, islamic, hate, christian, seeker, japan
unreliable|african, haitian, college, africa, dr, hill, un, dog, church, congolese
conspiracy|clinton, iran, fluoride, spring, religious, pp, natural, planned, mineral, parenthood
clickbait|trump, democrat, republican, fbi, featured, california, senate, crime, clinton, attorney
satire|christian, contact, im, grey, dont, jake, movie, thats, barista, he
fake|nuclear, blockchain, planet, experimentation, earth, headline, federation, stock, search, bitcoin
reliable|christian, church, religious, god, abortion, speech, faith, page, weapon, faction
bias|rose, abortion, planned, museum, prolife, democrat, protection, infanticide, voting, promising
political|god, word, wonder, grace, jesus, claim, human, presence, theological, eclipse
junksci|lazar, shapr, networking, ufo, base, individual, topic, facility, clearance

Word clouds are generated for different word sets.
![bag-of-word](EDA/bow.png)
Also, the following word clouds shows a few example of the most frequent words from different types of news.

A tendency of mentioning 'nuclear', 'blockchain' are appeared in 'fake' news. 
Word cloud for news labeled as 'Fake'
![fake_words](EDA/fake_words.png)
A tendency of mentioning 'muslin', 'islamic' are appeared in 'fake' news. 
Word cloud for news labeled as 'Hate':
![fake_words](EDA/hate_words.png)
A tendency of mentioning 'wave', 'theory' are appeared in 'fake' news. 
Word cloud for news labeled as 'Rumor':
![rumor_words](EDA/rumor_words.png)

## The Detector
The Fake News Detector is a deep learning model trained using the [Fake News Corpus](https://github.com/several27/FakeNewsCorpus) dataset. At this stage, the detector is able to perform fake and non-fake news detection. 
#### Pipeline
1. Fork and clone this repo to your local machine
2. Run the [script](src/model.py) to load, clean, tokenize the dataset, and to train the final model. Or you can load the trained model from [here](s3://fakenewscorpus), an AWS S3 bucket created for this project.
3. Test your own news articles using src/model.py and compare the prediction to your labeled news.

#### Text Preprocess
TF-IDF matrix was generated to train most of the models including MLP model. For LSTM model, the traditional TF-IDF matrix cannot be utilized since it doesn't obtain the original order of words in a sentence. Thus the build-in Tokenizer in Keras was used to tokenize text and transfor text to sequences.

#### Modeling and Model Comparison
Both classic machine learning methods, and the most advanced deep learning methods are utilized to build the fake news detector model.

Accuracy, AUROC, and f1-score were used for model comparison. 
ROC and Precision/Recall Curve plot is shown below is comparing among classic machine learning models including a Logistic Regression model, Multinormial Naive Bayes model, and a Gradient Boosting Model.
![ROC](img/ROC3.png)

By looking at ROC curve, the gradient boosting model appears to be a better model comparing to other method. 

As for deep learning models, the better model appeared to be the MLP model. The LSTM model is built without a embedding layer, therefore the performance is not as good as the MLP model.

**Metric**|**MLP**|**LSTM**|**Gradient Boosting**
----------|-------|--------|---------------------
Binary Accuracy|0.92|0.69|0.89
AUROC|0.94|0.69|0.91
F1-Score|0.89|0.30|0.87

## Future Direction
There are two major steps to be taken in the future:
* Improve the model by implement word embedding layer in LSTM model.
* Based on the detector, a fake news generator can be bring up on to schedule. Click for more update on the [Fake News Generator](https://github.com/hbbhsy/Fake-News-Generator).





