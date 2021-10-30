import pandas as pd
import zipfile

zf = zipfile. ZipFile('C:/Users/Dell/Desktop/Data Scientist Exercise.zip')
# if you want to see all files inside zip folder.
zf. namelist()
df =pd.read_csv(zf.open('galytix-interview-01/data/words.csv'))
df1 =pd.read_csv(zf.open('galytix-interview-01/data/ml.csv'))
df2 =pd.read_csv(zf.open('galytix-interview-01/data/phrases.csv'))

train_set = brown.sents()[:10000]
model = gensim.models.Word2Vec(train_set)

model.save('brown.embedding')
new_model = gensim.models.Word2Vec.load('brown.embedding')

len(new_model['university'])
100

new_model.wv.similarity('university','school') > 0.3
True

from nltk.data import find
word2vec_sample = str(find('models/word2vec_sample/pruned.word2vec.txt'))
model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_sample, binary=False)


len(model.vocab)
43981

len(model['university'])
300

model.most_similar(positive=['university'], topn = 3)
[('universities', 0.70039), ('faculty', 0.67809), ('undergraduate', 0.65870)]


model.doesnt_match('breakfast cereal dinner lunch'.split())
'cereal'



model.most_similar(positive=['woman','king'], negative=['man'], topn = 1)
[('queen', 0.71181)]

model.most_similar(positive=['Paris','Germany'], negative=['Berlin'], topn = 1)
[('France', 0.78840)]

import numpy as np
labels = []
count = 0
max_count = 1000
X = np.zeros(shape=(max_count,len(model['university'])))

for term in model.vocab:
    X[count] = model[term]
    labels.append(term)
    count+= 1
    if count >= max_count:
        break

# It is recommended to use PCA first to reduce to ~50 dimensions
from sklearn.decomposition import PCA
pca = PCA(n_components=50)
X_50 = pca.fit_transform(X)

# Using TSNE to further reduce to 2 dimensions
from sklearn.manifold import TSNE
model_tsne = TSNE(n_components=2, random_state=0)
Y = model_tsne.fit_transform(X_50)

# Show the scatter plot
import matplotlib.pyplot as plt
plt.scatter(Y[:,0], Y[:,1], 20)

# Add labels
for label, x, y in zip(labels, Y[:, 0], Y[:, 1]):
    plt.annotate(label, xy = (x,y), xytext = (0, 0), textcoords = 'offset points', size = 10)

    plt.show()

from gensim.models.word2vec import Word2Vec
# Load the binary model
model = Word2Vec.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary = True);

# Only output word that appear in the Brown corpus
from nltk.corpus import brown
words = set(brown.words())
print (len(words))

# Output presented word to a temporary file
out_file = 'pruned.word2vec.txt'
f = open(out_file,'wb')

word_presented = words.intersection(model.vocab.keys())
f.write('{} {}\n'.format(len(word_presented),len(model['word'])))

for word in word_presented:
    f.write('{} {}\n'.format(word, ' '.join(str(value) for value in model[word])))

f.close()
