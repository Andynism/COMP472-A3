import csv
import math

DELTA = 0.01

class NaiveBayesClassifier:
    def __init__(self, dataset, filtered):
        tsv_file = csv.reader(open(dataset, encoding="mbcs"), delimiter="\t")
        self.vocabulary = []
        self.filtered = filtered

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
        self.probability_dictionary()
    
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
    
    def probability_dictionary(self):
        skipped = []
        for word in self.vocabulary:
            if(self.filtered and self.fake_news[word] + self.real_news[word] == 1):
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
    
    def evaluate_tweets(self, dataset):
        tsv_file = csv.reader(open(dataset, encoding="mbcs"), delimiter="\t")
        ofilename = "trace_NB-BOW-FV.txt" if self.filtered else "trace_NB-BOW-OV.txt"
        output = open(ofilename, 'w')
        for row in tsv_file:
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
            output.write(f'{row[0]}  {prediction}  {score}  {row[2]}  {correct}\n')
    
    def compute_metrics(self):
        tracefile = "trace_NB-BOW-FV.txt" if self.filtered else "trace_NB-BOW-OV.txt"
        ofilename = "eval_NB-BOW-FV.txt" if self.filtered else "eval_NB-BOW-OV.txt"
        trace = open(tracefile, 'r')
        output = open(ofilename, 'w')

        count_correct = 0
        count_wrong = 0

        true_positives_real = 0
        false_positives_real = 0
        false_negatives_real = 0

        true_positives_fake = 0
        false_positives_fake = 0
        false_negatives_fake = 0

        for row in trace:
            entries = row.split("  ")
            if(entries[4] == 'correct\n'):
                count_correct += 1
            else:
                count_wrong += 1
            
            if(entries[1] == 'yes'):
                if(entries[3] == 'yes'):
                    true_positives_real += 1
                else:
                    false_positives_real += 1
                    false_negatives_fake += 1
            else:
                if(entries[3] == 'no'):
                    true_positives_fake += 1
                else:
                    false_positives_fake += 1
                    false_negatives_real += 1
        
        accuracy = "{:.4}".format(count_correct / (count_wrong + count_correct))
        output.write(f'{accuracy}\n')

        precision_real = true_positives_real/(true_positives_real + false_positives_real)
        precision_fake = true_positives_fake/(true_positives_fake + false_positives_fake)
        output.write(f'{"{:.4}".format(precision_real)}  {"{:.4}".format(precision_fake)}\n')

        
        recall_real = true_positives_real/(true_positives_real + false_negatives_real)
        recall_fake = true_positives_fake/(true_positives_fake + false_negatives_fake)
        output.write(f'{"{:.4}".format(recall_real)}  {"{:.4}".format(recall_fake)}\n')

        f1_measure_real = 2 * precision_real * recall_real / (precision_real + recall_real)
        f1_measure_fake = 2 * precision_fake * recall_fake / (precision_fake + recall_fake)
        output.write(f'{"{:.4}".format(f1_measure_real)}  {"{:.4}".format(f1_measure_fake)}\n')
        
print("Naive Bayes Classifier")
print("Original Vocabulary...")
original_vocabulary = NaiveBayesClassifier('./datasets/covid_training.tsv', False)
original_vocabulary.evaluate_tweets('./datasets/covid_test_public.tsv')
original_vocabulary.compute_metrics()
print("Filtered Vocabulary...")
filtered_vocabulary = NaiveBayesClassifier('./datasets/covid_training.tsv', True)
filtered_vocabulary.evaluate_tweets('./datasets/covid_test_public.tsv')
filtered_vocabulary.compute_metrics()
print("Success!")