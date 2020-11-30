import csv

class Vocabulary:
    def __init__(self, dataset, filtered):
        self.real_news = {}
        self.fake_news = {}
        self.real_news_size = 0
        self.fake_news_size = 0
        tsv_file = csv.reader(open(dataset, encoding="mbcs"), delimiter="\t")
        for row in tsv_file:
            if(row[2] == 'yes'):
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


print("------------------------ORIGINAL VOCABULARY------------------------")
original_vocabulary = Vocabulary('./datasets/covid_training.tsv', False)
print("Real News")
#print(original_vocabulary.real_news)
print(original_vocabulary.real_news_size)
print()
print("FAKE NEWS")
#print(original_vocabulary.fake_news)
print(original_vocabulary.fake_news_size)


print()

print("------------------------FILTERED VOCABULARY------------------------")
filtered_vocabulary = Vocabulary('./datasets/covid_training.tsv', True)
print("Real News")
#print(filtered_vocabulary.real_news)
print(filtered_vocabulary.real_news_size)
print()
print("FAKE NEWS")
#print(filtered_vocabulary.fake_news)
print(filtered_vocabulary.fake_news_size)