class NaiveBayesClassifier:
    def __init__(self, vocabulary, smoothing, log):
        self.vocabulary = vocabulary
        self.smoothing = smoothing
        self.log = log