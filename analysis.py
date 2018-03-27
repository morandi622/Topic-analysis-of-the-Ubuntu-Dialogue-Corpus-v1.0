
''''
REPLACE THE BELOW PATH TO YOUR DIRECTORY
'''
%cd "\\vmware-host\Shared Folders\Shared Folders\Desktop\4"



import glob,csv,spacy, nltk, numpy as np, matplotlib.pyplot as plt, string, re
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.naive_bayes import MultinomialNB


text=[]
for file in glob.glob('*.tsv'):
    with open(file) as tsvfile:
        tsvfile = csv.reader(tsvfile, delimiter='\t')
        _lst=''
        for row in tsvfile:
            _lst+=(row[-1]+" ")
    text.append(_lst)


en_nlp = spacy.load('en') #we use Spacy to perform lemmatization
old_tokenizer = en_nlp.tokenizer
en_nlp.tokenizer = lambda string: old_tokenizer.tokens_from_list(re.compile('(?u)\\b\\w\\w+\\b').findall(string)) # replace the Spacy tokenizer with the regexp used in CountVectorizer

def custom_tokenizer(document): # create a custom tokenizer using the SpaCy document processing pipeline for lemmatization
    doc_spacy = en_nlp(document, entity=False, parse=False)
    return [token.lemma_ for token in doc_spacy]

stop_words = nltk.corpus.stopwords.words('english') + \
             ['--','\'s','\'re', u'â€”','``',"''","-","..","...",";-)","=","-PRON-"] + \
list(string.punctuation) #defining stop words

# LDA being a probabilistic graphical model only requires raw counts, so thats why we used  CountVectorizer below
lemma_vect = CountVectorizer(tokenizer=custom_tokenizer, min_df=15,stop_words=stop_words, max_features=10000, max_df=0.95) # define a count vectorizer with the custom tokenizer and eliminate stop_words. We’ll remove words that dont appear in at least min_df percent of the documents, and we’ll limit the bag-of-words model to the max_features words that are most common after removing the bottom min_df words and the top max_df percent
tf= lemma_vect.fit_transform(text) # transform text using CountVectorizer with Spacy lemmatization
feature_names = np.array(lemma_vect.get_feature_names()) #name of features

# A tf-idf transformer is applied to the bag of words matrix that NMF must process with the TfidfVectorizer.
tfidf_vectorizer = TfidfVectorizer(vocabulary=feature_names, tokenizer=custom_tokenizer, stop_words=stop_words) # we used the vocabulary from CountVectorizer
tfidf = tfidf_vectorizer.fit_transform(text)
nmf = NMF(n_components=50, alpha=.1, l1_ratio=.5, init='nndsvda', random_state=42, beta_loss='kullback-leibler',solver='mu').fit(tfidf)


# LDA can only use raw term counts for LDA because it is a probabilistic graphical model, that's why we used term-frequency tf above
lda = LatentDirichletAllocation(n_components=50, learning_method="batch",max_iter=25, random_state=0) #Latent Dirichlet Allocation in batch mode and n_components = number of topic
document_topics = lda.fit_transform(tf)

selection_lda=lda.components_.sum(1).argsort()[::-1] # most popular topics (descending order) by pseudo-counts. Note: Since the complete conditional for topic word distribution is a Dirichlet, components_[i, j] can be viewed as pseudocounts that represents the number of times word j was assigned to topic i. It can also be viewed as distribution over the words for each topic after normalization: model.components_ / model.components_.sum(axis=1)[:, np.newaxis]
selection_nmf=nmf.components_.sum(1).argsort()[::-1]


n_top_words = 8 #top words per each topic  -we limit to the 10 most important topics

def display_topics(model, feature_names, no_top_words,selection):

    for topic_idx, topic in enumerate(model.components_[selection]):
        print("Topic {}: {}  ").format(topic_idx," ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]]))

display_topics(lda, feature_names, n_top_words, selection_lda[:10])
display_topics(nmf, feature_names, n_top_words, selection_nmf[:10])




def plot_topics(model,selection):
    plt.bar(np.arange(10), model.components_.sum(1)[selection][:10])
    sorting = np.argsort(model.components_, axis=1)[:,::-1]  # For each topic (a row in the components_), sort the features (ascending order)
    topic_names = ["{:>2} ".format(i) + " ".join(words) for i, words in enumerate(feature_names[sorting[selection, :2]])]
    _=plt.xticks(np.arange(10),topic_names[:10],rotation=45)
    plt.ylabel('word\'s pseudo-counts')
    plt.xlabel('topics')
    plt.title(model.__class__.__name__+' topic modelling')
    plt.tight_layout()



plot_topics(lda,selection_lda)
plot_topics(nmf,selection_nmf)








# The second programming task is to write a classifier (or topic detector) , so given a random conversation (.tsv file), it will generate a set of relevant topics mentioned


lemma_vect = CountVectorizer(tokenizer=custom_tokenizer, min_df=15,stop_words=stop_words, max_features=10000, max_df=0.95)
X_train=training_data= lemma_vect.fit_transform(text[:100000])
feature_names = np.array(lemma_vect.get_feature_names()) #name of features

lemma_vect2 = CountVectorizer(tokenizer=custom_tokenizer, min_df=15,stop_words=stop_words, max_features=10000, max_df=0.95, vocabulary=feature_names)
X_test=testing_data= lemma_vect2.fit_transform(text[100000:200000])
lda = LatentDirichletAllocation(n_components=10, learning_method="batch",max_iter=25, random_state=0)
lda.fit(training_data)
training_features = lda.transform(training_data)
testing_features = lda.transform(testing_data)
y_train=training_features.argmax(1)
y_test=testing_features.argmax(1)

mn=MultinomialNB()
mn.fit(X_train, y_train)
mn.score(X_test,y_test)








## PageRank ordering of topics

cosine_sim=np.dot(document_topics.T, document_topics) #to compute the similarity between two topics we need to produce a term vector for each sentence and compute the dot product of the unit vectors for those documents. NLTK exposes the nltk.cluster.util.cosine_distance(v1,v2) function for computing cosine similarity
markov=cosine_sim/cosine_sim.sum(axis=1).reshape(-1,1) #stochastic matrix

d=0.02  # dumping factor. We need to make sure that the similarity matrix is always irreducible and aperiodic. To solve this problem, Page et al. (1998) suggest reserving some low probability for jumping to any node in the graph. This way the random walker can “escape” from periodic or disconnected components, which makes the graph irreducible and aperiodic. If we assign a uniform probability for jumping to any node in the graph, we are left with the following modified version of markov matrix, which is known as PageRank
pageRank=(1-d)*markov + d*np.ones_like(markov)/markov.shape[0]
stationary_distribution=np.linalg.matrix_power(pageRank,100)[0,:]   #Writing the page rank r for the row vector of rankings, the equation becomes r = r # PageRank. Hence r is the stationary distribution (stationary_distribution vector) of the stochastic matrix PageRank. Let’s think of PageRank(i,j) as the probability of “moving” from sentence i to sentence j. The value PageRank(i,j) has the interpretation P(i,j) = 1/k if i has k outbound links, and j is one of them. P(i,j) = 0 if i has no direct link to j. Thus, motion from a topic to another is that of a document surfer who moves by randomly (that is with equal probability) selecting one topic on that document.


selection_PageRank=stationary_distribution.argsort()[::-1]
display_topics(lda, feature_names, n_top_words, [selection_PageRank[:10]])
plot_topics(lda,selection_lda)







