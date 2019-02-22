from important_words import get_important_words


def get_bigrams(words):
	if len(words) < 2:
		return None
	bigrams = []
	for i in range(len(words) - 1):
		for j in range(i + 1, len(words)):
			bigrams.append([words[i], words[j]])
	return bigrams


def get_trigrams(words):
	if len(words) < 3:
		return None
	trigrams = []
	for i in range(len(words) - 2):
		for j in range(i + 1, len(words) - 1):
			for k in range(j + 1, len(words)):
				trigrams.append([words[i], words[j], words[k]])
	return trigrams


if __name__=='__main__':
	from sys import argv
	words = get_important_words(argv[1])
	print (words)
	print (get_bigrams(words))
	print (get_trigrams(words))
	print ("\n\n")	
