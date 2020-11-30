import copy
import vocabulary

class NaiveBayesClassifier:
    def __init__(self, vocabulary, smoothing, log):
        self.vocabulary = vocabulary
        self.smoothing = smoothing
        self.log = log
    
    def probability_per_word(self):
        for (key, value) in self.vocabulary.real_news:
            self.vocabulary.real_news[key] = log(value/self.vocabulary.real_news_size)
        
        for (key, value) in self.vocabulary.real_news:
            self.vocabulary.fake_news[key] = log(value/self.vocabulary.fake_news.size)

nbc = NaiveBayesClassifier(vocabulary('./datasets/covid_training.tsv', False))
print(nbc.vocabulary.real_news)
        

