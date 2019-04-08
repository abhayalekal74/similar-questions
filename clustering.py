from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
import sys
import re
from pprint import PrettyPrinter


def process(line):
	if line:
		line = line.lower()
		line = re.sub(r'<.*?>', ' ', line)
		line = re.sub(r'[^a-z0-9]', ' ', line)
		line = re.sub(r'\s+', ' ', line)
		line = line.strip()
		return line
	return None


docs = list()
pp = PrettyPrinter(indent = 4)


with open(sys.argv[1], 'r') as f:
	for line in f.readlines():
		line = process(line)
		if line:
			docs.append(process(line))


docs = docs[:100000]

vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(docs)

dbscan = DBSCAN(eps = 0.95, min_samples = 20) #, n_jobs = -1)
dbscan.fit(X)

label_sentence_map = dict()

for i in range(len(dbscan.labels_)):
	try:
		sentences = label_sentence_map[dbscan.labels_[i]]
	except:
		sentences = list()
		label_sentence_map[dbscan.labels_[i]] = sentences
	sentences.append(docs[i])


print ("Total Labels", len(set(dbscan.labels_)))

for i in sorted(set(dbscan.labels_)):
	print ("Label", i, len(label_sentence_map[i]))
	pp.pprint(label_sentence_map[i][:5])
