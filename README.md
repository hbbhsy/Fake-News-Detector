# Fake-News-Detector

![title](img/fakenews.jpg)

Today we are living in the Era of information explosion. Along with the development of the Internet, the emergence and widespread adoption of the social media concept have changed the way news is formed and published. 

News has become faster, less costly and easily accessible with social media. This change has come along with some disadvantages as well. In particular, beguiling content, such as fake news made by social media users, is becoming increasingly dangerous. The fake news problem, despite being introduced for the first time very recently, has become an important research topic due to the high content of social media. Writing fake comments and news on social media is easy for users. The main challenge is to determine the difference between real and fake news. 

## Scope
The focus of this work is to apple novel deep learning approaches to 

![info-explosion](img/Information-Explosion.jpg)

## Data

The [Fake News Corpus](https://github.com/several27/FakeNewsCorpus) is an open source dataset composed of millions of news articles mostly scraped from a curated list of 1001 domains from http://www.opensources.co/. 

The corpus was created by scraping (using scrapy) all the domains as provided by http://www.opensources.co/. Then all the pure HTML content was processed to extract the article text with some additional fields (listed below) using the newspaper library. Each article has been attributed the same label as the label associated with its domain.

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

#### Text Preprocess
TF-IDF matrix was generated to train models. A maximum of 10000 most frequent word were selected to construct TF-IDF.

For LSTM model, the traditional TF-IDF matrix cannot be utilized since it doesn't obtain the original order of words in a sentence. Thus the build-in Tokenizer in Keras was used to tokenize text and transfor text to sequences.

#### Modeling and Model Comparison
Both classic machine learning methods, and the most advanced deep learning methods are utilized to build the fake news detector model.




