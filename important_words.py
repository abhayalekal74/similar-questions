from pos_tag import get_tags
from nltk.corpus import stopwords
import re

IMP_WORD_LEN = 6
MIN_WORD_LEN = 1 

alpha_words = re.compile(r'^[a-zA-Z-]+$')
eng_stopwords = set(stopwords.words('english'))


def get_important_words(sentence):
	tags = get_tags(sentence)
	important_words = list(set([w[0] for w in tags if w[0] not in eng_stopwords and len(w[0]) > MIN_WORD_LEN and ((w[1].startswith('NN') or len(w[0]) >= IMP_WORD_LEN) and alpha_words.search(w[0]))]))
	important_words.sort()
	return important_words


if __name__=='__main__':
	from sys import argv
	print (argv[1])
	print (get_important_words(argv[1]))	
	print ("\n\n")
