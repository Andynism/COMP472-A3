import csv

fd = open('./datasets/covid_training.tsv', encoding="mbcs")
rd = csv.reader(fd, delimiter="\t")

vocabulary = {}
vocabulary_size = 0
for row in rd:
    for word in row[1].split():
        if(word in vocabulary):
            vocabulary[word] = vocabulary.get(word) + 1
        else: 
            vocabulary[word] = 1
            vocabulary_size += 1

print(vocabulary)
print(vocabulary_size)