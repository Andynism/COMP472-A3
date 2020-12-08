import csv
import math

DELTA = 0.01

class NaiveBayesClassifier:
    def __init__(self, dataset, filtered):
        tsv_file = csv.reader(open(dataset, encoding="mbcs"), delimiter="\t")
        self.vocabulary = []

        self.real_news = {}
        self.total_words_real = 0
        self.p_real_given_word = {}
        self.total_real_tweets = 0
        self.p_real = 0

        self.fake_news = {}
        self.total_words_fake = 0
        self.p_fake_given_word = {}
        self.total_fake_tweets = 0
        self.p_fake = 0

        self.create_vocabulary(tsv_file)
        self.probability_dictionary(filtered)
        self.evaluate_tweets(dataset, filtered)
    
    def create_vocabulary(self, tsv_file):
        for row in tsv_file:
            if(row[0] == 'tweet_id'):
                continue
            if(row[2] == 'yes'):
                self.total_real_tweets += 1
                for word in row[1].split():
                    if(word not in self.vocabulary):
                        self.vocabulary.append(word)
                        self.real_news[word] = 0
                        self.fake_news[word] = 0

                    self.real_news[word] += 1
                    self.total_words_real += 1

            elif (row[2] == 'no'):
                self.total_fake_tweets += 1
                for word in row[1].split():
                    if(word not in self.vocabulary):
                        self.vocabulary.append(word)
                        self.real_news[word] = 0
                        self.fake_news[word] = 0
                    
                    self.fake_news[word] += 1
                    self.total_words_fake += 1
            else:
                print(row)
    
    def probability_dictionary(self, filtered):
        skipped = []
        for word in self.vocabulary:
            if(filtered and self.fake_news[word] + self.real_news[word] == 1):
                skipped.append(word)
        
        for word in skipped:
            self.vocabulary.remove(word)
            if(self.fake_news[word] == 1):
                self.fake_news.pop(word)
                self.total_words_fake -= 1
            elif(self.real_news[word] == 1):
                self.real_news.pop(word)
                self.total_words_real -= 1

        for word in self.vocabulary:
            self.p_real_given_word[word] = (self.real_news[word] + DELTA) / (self.total_words_real + DELTA * len(self.vocabulary))
            self.p_fake_given_word[word] = (self.fake_news[word] + DELTA) / (self.total_words_fake + DELTA * len(self.vocabulary))

        total_tweets = self.total_real_tweets + self.total_fake_tweets
        self.p_real = self.total_real_tweets / total_tweets
        self.p_fake = self.total_fake_tweets / total_tweets
    
    def evaluate_tweets(self, dataset, filtered):
        tsv_file = csv.reader(open(dataset, encoding="mbcs"), delimiter="\t")
        ofilename = "trace_NB-BOW-FV.txt" if filtered else "trace_NB-BOW-OV.txt"
        output = open(ofilename, 'w')
        for row in tsv_file:
            if(row[0] == 'tweet_id'):
                continue
            real_news_score = 0
            fake_news_score = 0

            for word in row[1].split():
                if(word in self.p_real_given_word):
                    real_news_score += self.p_real_given_word.get(word)

                if(word in self.p_fake_given_word):
                    fake_news_score += self.p_fake_given_word.get(word)

            prediction = "yes"
            score = real_news_score
            if(fake_news_score > real_news_score):
                prediction = "no"
                score = fake_news_score
            score = "{:.2e}".format(score)

            correct = "correct" if prediction == row[2] else "wrong"
            # Print one line of output
            output.write(f'{row[0]}  {prediction}  {score}  {row[2]}  {correct}')


# print("------------------------ORIGINAL VOCABULARY------------------------")
# original_vocabulary = Vocabulary('./datasets/covid_training.tsv', False)
# print("Real News")
# #print(original_vocabulary.real_news)
# print(original_vocabulary.total_words_real)
# print()
# print("FAKE NEWS")
# #print(original_vocabulary.fake_news)
# print(original_vocabulary.total_words_fake)


# print()

print("Doing Naive Bayes Classifier")
filtered_vocabulary = NaiveBayesClassifier('./datasets/covid_training.tsv', True)
print("Success!")