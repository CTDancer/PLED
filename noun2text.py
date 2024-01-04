import os

with open('noun_corpus.txt', 'r') as f1:
    with open('text_corpus.txt', 'w') as f2:
        for word in f1:
            word = "A photo of " + word
            f2.write(word)