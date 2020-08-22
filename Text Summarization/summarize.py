

from nltk.corpus import stopwords


f = open(file_name, "r")
lines = f.readlines()
article = lines[0].split(". ")
stop_words = stopwords.words('english')
summarize_text = []
sentences = []
for sentence in article:
    print(sentence)
    sentences.append(sentence.replace("[^a-zA-Z]", " ").split(" "))
    sentences.pop()


# to be continued!
