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

print "Importing Emails"
email = mailing_list_parser.get_messages()
email = mailing_list_parser.strip_footer(email)
roots = mailing_list_parser.get_root(email)

messages = []

print "Calculating Sentiment"
for root in tqdm.tqdm(roots):
    blob = TextBlob(root.message)
    sentiment = blob.sentiment.polarity
    messages.append((sentiment,root))

print 'Grouping threads'
thread_dict = mailing_list_parser.bundle_email(email)

print "Converting to corpus"
#generate a corpus from the message subject lines
subjects = []

for m in messages:
    if m[1].message == None:#change back to .subject
        subjects.append('')
    else:
        subjects.append(m[1].message)

print "\tTokenizing corpus"
tokens = []
for i,s in enumerate(subjects):
    #print i
    tokens.append(word_tokenize(s))
subjects = tokens
stopset = set(stopwords.words('english'))
subjects = [[w for w in words if w not in stopset] for words in subjects]
subjects = [[w for w in words if w not in string.punctuation] for words in subjects]
subjects = [[w for w in words if w not in string.whitespace] for words in subjects]
subjects = [[w for w in words if w not in string.digits] for words in subjects]
subjects = [[w for w in words if not w=='OpenSCAD' ] for words in subjects]

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

print "Applying LDA model"
model = models.LdaModel(corpus, id2word=dictionary, num_topics=50)
#print model.print_topics()
#plt.hist([s[0] for s in messages],100)
#plt.show()

print "Calculating sentiments"
print "\tGrouping documents"
doc_groups = defaultdict(set)
for i in tqdm.tqdm(range(0, len(messages))):
    topics = model[corpus[i]]
    for t in topics:
        doc_groups[t[0]].add(i)
print "\tCalculating group sentiments"
group_sents = {}
for g in doc_groups.keys():
    d_list = list(doc_groups[g])
    d_list = [messages[i][0] for i in d_list]
    if len(d_list)>1:
        group_sents[g] = (mean(d_list),pstdev(d_list),len(d_list))
    else:
        group_sents[g] = (mean(d_list),0.0,len(d_list))
gs = doc_groups.keys()
gs.sort()
print "Found {} groups (average {} documents in each)".format(len(gs),len(gs)/len(messages))
#print "Group\tSentiment\tStdev\tCount"
#for g in gs:
#    d = group_sents[g]
#    print "{}\t{}\t{}\t{}".format(g, d[0],d[1],d[2])

gs_neg = gs[:]
gs_neg.sort(key=lambda i:group_sents[i][0])
for i in gs_neg[:10]:
    print model.show_topic(i)
