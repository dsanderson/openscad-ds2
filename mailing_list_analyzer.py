import gensim, tqdm, string
from gensim import corpora, models, similarities
from textblob import TextBlob
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import mailing_list_parser
import matplotlib.pyplot as plt
from collections import defaultdict

def mean(data):
    """Return the sample arithmetic mean of data."""
    n = len(data)
    if n < 1:
        raise ValueError('mean requires at least one data point')
    return sum(data)/n # in Python 2 use sum(data)/float(n)

def _ss(data):
    """Return sum of square deviations of sequence data."""
    c = mean(data)
    ss = sum((x-c)**2 for x in data)
    return ss

def pstdev(data):
    """Calculates the population standard deviation."""
    n = len(data)
    if n < 2:
        raise ValueError('variance requires at least two data points')
    ss = _ss(data)
    pvar = ss/n # the population variance
    return pvar**0.5

def text2bow(text):
    ''

def fetch_emails():
    email = mailing_list_parser.get_messages()
    email = mailing_list_parser.strip_footer(email)
    roots = mailing_list_parser.get_root(email)
    return email, roots

def create_corpus(email):
    messages = []
    subjects = []

    print "\tExtracting email text"
    for e in email:
        if e.subject == None:
            txt = ' '
        else:
            txt = e.subject + ' '
        #if e.message != None:
        #    txt = txt + e.message
        subjects.append(txt)

    subject_copy = subjects[:]
    print "\tTokenizing corpus"
    tokens = []
    for i,s in enumerate(subjects):
        #print i
        tokens.append(word_tokenize(s))
    subjects = tokens
    subjects = subjects = [[w.lower() for w in words] for words in subjects]
    stopset = set(stopwords.words('english'))
    subjects = [[w for w in words if w not in stopset] for words in subjects]
    subjects = [[w for w in words if w not in string.punctuation] for words in subjects]
    subjects = [[w for w in words if w not in string.whitespace] for words in subjects]
    subjects = [[w for w in words if w not in string.digits] for words in subjects]
    subjects = [[w for w in words if not w=='openscad' ] for words in subjects]

    bespoke_set = set(['i','would','http','wrote','https',"'s","n't","''","``","//"])
    subjects = [[w for w in words if w not in bespoke_set ] for words in subjects]

    print "\tRemoving singletons"
    frequency = defaultdict(int)
    for subject in subjects:
        for token in subject:
            #print token
            frequency[token] += 1
    subjects = [[w for w in tokens if frequency[w]>1] for tokens in subjects]

    print "\tGenerating dictionary"
    dictionary = gensim.corpora.Dictionary(subjects)
    corpus = [dictionary.doc2bow(s) for s in subjects]
    return corpus, dictionary, subjects, subject_copy

def gen_model(corpus, dictionary, n_topics):
    #model = models.LdaModel(corpus, id2word=dictionary, num_topics=n_topics)
    #model = models.ldamulticore.LdaMulticore(corpus, id2word=dictionary, num_topics=n_topics)
    model = models.hdpmodel.HdpModel(corpus, id2word=dictionary)
    model.optimal_ordering()
    return model

def group_docs(corpus, model):
    doc_groups = defaultdict(set)
    for i in tqdm.tqdm(range(0, len(subjects))):
        topics = model[corpus[i]]
        for t in topics:
            doc_groups[t[0]].add(i)
    return doc_groups

def gen_reports(doc_groups, model, dictionary, n_topics = None):
    #dk = doc_groups.keys()
    #dk.sort(key=lambda i:len(doc_groups[i]))
    #dk.reverse()
    topics = model.show_topics(-1, formatted=False)
    if n_topics == None:
        n_topics = len(topics)
    print "Top {} topics, by HDP ordering:".format(n_topics)
    for topic in topics[:n_topics]:
        terms = topic[1][:5]
        print_terms = ', '.join([t[0] for t in terms])
        print "Topic {}: {}".format(topic[0],print_terms)

def dig(topic, doc_groups, docs):
    print "Contents of topic {}:".format(topic)
    for i in doc_groups[topic]:
        print '\t'+docs[i]

def get_topic_sentiments(doc_groups, emails):
    topics = doc_groups.keys()
    sentiments = []
    for topic in topics:
        sents = []
        for doc in doc_groups[topic]:
            txt = emails[doc].message
            blob = TextBlob(txt)
            sents.append(blob.sentiment.polarity)
        m = mean(sents)
        if len(sents>1):
            sd = pstdev(sents)
        sentiments.append((topic, m, sd, len(sents)))
    return sentiments

print "Importing Emails"
email, roots = fetch_emails()
#email = email[:100]
print "{} emails".format(len(email))

print "Converting to corpus"
#generate a corpus from the message subject lines
corpus, dictionary, subjects, subject_copy = create_corpus(email)

print "Applying LDA model"
model = gen_model(corpus, dictionary, 100)

print "\tGrouping documents"
doc_groups = group_docs(corpus, model)
sentiments = get_topic_sentiments(doc_groups, email)
gen_reports(doc_groups, model, dictionary, sentiments)

while True:
    topic = int(raw_input("Topic ID:"))
    dig(topic, doc_groups, subject_copy)
