import nltk


def get_tags(sentence):
	return nltk.pos_tag(nltk.word_tokenize(sentence))


if __name__=='__main__':
	from sys import argv
	print (get_tags(argv[1]))
