import csv

class Vocabulary:
    def __init__(self, dataset, filtered):
        self.vocabulary_dictionnary = {}
        self.vocabulary_size = 0
        tsv_file = csv.reader(open(dataset, encoding="mbcs"), delimiter="\t")
        for row in tsv_file:
            for word in row[1].split():
                if(word in self.vocabulary_dictionnary):
                    self.vocabulary_dictionnary[word] = self.vocabulary_dictionnary.get(word) + 1
                else: 
                    self.vocabulary_dictionnary[word] = 1
                    self.vocabulary_size += 1

        if(filtered):
            keys_to_delete = []
            for (key,value) in self.vocabulary_dictionnary.items():
                if(value == 1):
                    keys_to_delete.append(key)
                    self.vocabulary_size -= 1
            
            for key in keys_to_delete:
                del self.vocabulary_dictionnary[key]

v = Vocabulary('./datasets/covid_training.tsv', True)
print(v.vocabulary_dictionnary)
print(v.vocabulary_size)


