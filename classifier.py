import numpy as np
import string
import re

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from pathlib import Path


# remove punctuation with regex
def rm_punctuation(text: str) -> str:
    regex = re.compile('[%s]' % re.escape(string.punctuation))
    return regex.sub("", text)


def txt_normalize(text: str) -> list[str]:
    stop_list = stopwords.words('english')
    ps = PorterStemmer()
    text = rm_punctuation(text)

    # Tokenization
    tokens = [token.lower() for token in word_tokenize(text) if len(token) > 1]
    tokens = [tok for tok in tokens if tok not in stop_list]
    stems = [ps.stem(t) for t in tokens]

    return stems


def create_bow(files: list[str]) -> dict:
    bow: dict = {}

    for doc in files:
        with open(doc, 'r+') as f:
            stems: list[str] = txt_normalize(f.read())

            for s in stems:
                if s not in bow.keys():
                    bow[s] = 1
                else:
                    bow[s] += 1

    return bow


def word_log_likelihoods(bow: dict, vocab: dict) -> dict:
    total_words = len(vocab.keys())
    total_occurrences = sum(bow.values())

    loglikelihood = {}
    for word in vocab.keys():
        class_occurrences = 0 if word not in bow.keys() else bow[word]
        loglikelihood[word] = np.log((class_occurrences + 1) / (total_occurrences + total_words))

    return loglikelihood


class Classifier:

    def __init__(self, training_dir: str):

        print("Fetch training data... :", end=" ")
        training_dir_path = Path(training_dir)

        train_geo_docs = list(training_dir_path.joinpath('geo').glob('*.txt'))
        train_non_geo_docs = list(training_dir_path.joinpath('non_geo').glob('*.txt'))

        print('DONE.')

        print("Create VOCABULARY... :", end=" ")

        self.geo_bow = create_bow([str(doc) for doc in train_geo_docs])
        self.non_geo_bow = create_bow([str(doc) for doc in train_non_geo_docs])
        self.vocab = self.geo_bow.copy()

        for word in self.non_geo_bow.keys():
            if word not in self.vocab.keys():
                self.vocab[word] = self.non_geo_bow[word]
            else:
                self.vocab[word] += self.non_geo_bow[word]

        print("DONE.")

        print('Calculating the probabilities of classes:')

        self.geo_p = np.log(len(train_geo_docs)
                            / (len(train_geo_docs) + len(train_non_geo_docs)))

        self.non_geo_p = np.log(len(train_non_geo_docs)
                                / (len(train_geo_docs) + len(train_non_geo_docs)))
        print(f'--Previous log probability of GEO docs: {self.geo_p}')
        print(f'--Previous log probability of non-GEO docs: {self.non_geo_p}')
        self.geo_loglikelihood = word_log_likelihoods(self.geo_bow, self.vocab)
        self.non_geo_loglikelihood = word_log_likelihoods(self.non_geo_bow, self.vocab)
        print('DONE.')

    """
    All the documents within the specified directory are categorized by it.
     The vocabulary of all the Training and the bag of words and likelihoods for each 
    class are used to carry out the classification.
    """

    def classify(self, test_dir: str) -> tuple:
        print("Fetching test data... :", end="")

        test_dir_path = Path(test_dir)

        test_geo_docs = list(test_dir_path.joinpath('geo').glob('*.txt'))
        test_non_geo_docs = list(test_dir_path.joinpath('non-geo').glob('*.txt'))
        test_docs = test_geo_docs + test_non_geo_docs

        gold_labels = np.concatenate([np.zeros(len(test_geo_docs)), np.ones(len(test_non_geo_docs))])
        system_labels = np.zeros(len(test_docs))
        print('DONE.')

        print("Document classifying...", end="")

        for i in range(len(test_docs)):
            pp = [self.geo_p, self.non_geo_p]

            with open(test_docs[i], 'r+') as file:
                words = txt_normalize(file.read())

                for w in words:
                    if w in self.vocab.keys():
                        pp[0] += self.geo_loglikelihood[w]
                        pp[1] += self.non_geo_loglikelihood[w]

                system_labels[i] = np.argmax(pp)

        print("DONE.")
        return system_labels, gold_labels







