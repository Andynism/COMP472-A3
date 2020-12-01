import csv
import math

class NaiveBayesClassifier:
    def __init__(self, dataset, filtered):
        tsv_file = csv.reader(open(dataset, encoding="mbcs"), delimiter="\t")
        self.real_news = {}
        self.real_news_size = 0
        self.real_news_probability = {}
        self.real_news_training_size = 0

        self.fake_news = {}
        self.fake_news_size = 0
        self.fake_news_probability = {}
        self.fake_news_training_size = 0

        self.create_vocabulary(filtered, tsv_file)
        self.probability_dictionary()
        self.evaluate_tweets(dataset)
    
    def create_vocabulary(self, filtered, tsv_file):
        for row in tsv_file:
            if(row[2] == 'yes'):
                self.real_news_training_size += 1
                for word in row[1].split():
                    if(word in self.real_news):
                        self.real_news[word] = self.real_news.get(word) + 1
                    else: 
                        self.real_news[word] = 1
                        self.real_news_size += 1

                if(filtered):
                    keys_to_delete = []
                    for (key,value) in self.real_news.items():
                        if(value == 1):
                            keys_to_delete.append(key)
                            self.real_news_size -= 1
                    
                    for key in keys_to_delete:
                        del self.real_news[key]
                

            elif (row[2] == 'no'):
                self.fake_news_training_size += 1
                for word in row[1].split():
                    if(word in self.fake_news):
                        self.fake_news[word] = self.fake_news.get(word) + 1
                    else: 
                        self.fake_news[word] = 1
                        self.fake_news_size += 1
                
                if(filtered):
                    keys_to_delete = []
                    for (key,value) in self.fake_news.items():
                        if(value == 1):
                            keys_to_delete.append(key)
                            self.fake_news_size -= 1
                    
                    for key in keys_to_delete:
                        del self.fake_news[key]
            else:
                print(row)
    
    def probability_dictionary(self):
        for (key, value) in self.real_news.items():
            self.real_news_probability[key] = math.log(value/self.real_news_size)
        
        for (key, value) in self.fake_news.items():
            self.fake_news_probability[key] = math.log(value/self.fake_news_size)
    
    def evaluate_tweets(self, dataset):
        tsv_file = csv.reader(open(dataset, encoding="mbcs"), delimiter="\t")
        self.total_size = self.fake_news_training_size + self.real_news_training_size
        for row in tsv_file:
            real_news_score = 0
            fake_news_score = 0

            for word in row[1].split():
                if(word in self.real_news_probability):
                    real_news_score += self.real_news_probability.get(word)

                if(word in self.fake_news_probability):
                    fake_news_score += self.fake_news_probability.get(word)

            if(fake_news_score < real_news_score):
                print("REAL NEWS", row[2])
            else:
                print("FAKE NEWS", row[2])


# print("------------------------ORIGINAL VOCABULARY------------------------")
# original_vocabulary = Vocabulary('./datasets/covid_training.tsv', False)
# print("Real News")
# #print(original_vocabulary.real_news)
# print(original_vocabulary.real_news_size)
# print()
# print("FAKE NEWS")
# #print(original_vocabulary.fake_news)
# print(original_vocabulary.fake_news_size)


# print()

print("------------------------FILTERED VOCABULARY------------------------")
filtered_vocabulary = NaiveBayesClassifier('./datasets/covid_training.tsv', True)
print("Real News")
#print(filtered_vocabulary.real_news)
print(filtered_vocabulary.real_news_size)
print()
print("FAKE NEWS")
#print(filtered_vocabulary.fake_news)
print(filtered_vocabulary.fake_news_size)