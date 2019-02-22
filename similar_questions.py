from ngrams import get_trigrams, get_bigrams
from important_words import get_important_words
from difflib import SequenceMatcher
from time import time
from nltk.corpus import stopwords
import json
from os.path import join

SIMILAR_QUESTION_THRES = 0.65
BIGRAMS_SYNC = "bigrams.map"
TRIGRAMS_SYNC = "trigrams.map"
QUESTIONS_SYNC = "questions.map"

bigrams_map, trigrams_map, questions_map = None, None, None

stopwords = set(stopwords.words('english'))


def add_to_map(src_map, keys, val):
	for key in keys:
		key = str(key)
		try:
			vals = src_map[key]
		except KeyError as e:
			vals = list()
			src_map[key] = vals
		vals.append(val)


def get_val(src_map, key):
	key = str(key)
	try:
		return src_map[key]
	except KeyError as e:
		return None


def get_values(src_map, keys):
	vals = list()
	if keys:
		for key in keys:
			val = get_val(src_map, key)
			if val:
				vals += val
	return vals


def get_ngrams(question):
	words = get_important_words(question)
	bigrams = get_bigrams(words)
	trigrams = get_trigrams(words)
	return bigrams, trigrams


def print_vals(key, vals):
	print ("\n\n", key)
	for v in vals:
		print (v)


def get_similar_questions(question):
	bigrams, trigrams = get_ngrams(question)
	similar_questions = get_values(trigrams_map, trigrams)
	if not similar_questions:
		similar_questions = get_values(bigrams_map, bigrams)
	return set(similar_questions)


def get_simplified_sentence(sentence):
	words = [w for w in sentence.split() if w not in stopwords]
	words.sort()
	return ' '.join(words)


def rank_questions(question, similar_questions):
	ranked_questions = list()
	question = get_simplified_sentence(question)
	for sq in similar_questions:
		similarity = SequenceMatcher(None, question, get_simplified_sentence(sq)).ratio()
		if similarity >= SIMILAR_QUESTION_THRES:
			ranked_questions.append([similarity, sq])
	ranked_questions.sort(key = lambda x: x[0], reverse = True)
	return ranked_questions


def write_to_file(src_dir, sync, src_map):
	with open(join(src_dir, sync), 'w') as f:
		f.write(json.dumps(src_map))


def read_from_file(src_dir, sync):
	with open(join(src_dir, sync), 'r') as s:
		return json.loads(s.read()) 
	

def load_stored_map(src_dir):
	try:
		bigrams_map = read_from_file(src_dir, BIGRAMS_SYNC) 
		trigrams_map = read_from_file(src_dir, TRIGRAMS_SYNC) 
		questions_map = read_from_file(src_dir, QUESTIONS_SYNC) 
		print ("Maps loaded from src_dir")
	except FileNotFoundError:
		return dict(), dict(), dict()
	return bigrams_map, trigrams_map, questions_map
	

def train(data_file, src_dir):
	import progressbar

	global bigrams_map, trigrams_map, questions_map
	progressbar.streams.wrap_stderr()

	print ("Starting training...")

	bigrams_map, trigrams_map, questions_map = load_stored_map(src_dir)

	if not bigrams_map or not trigrams_map or not questions_map:
		print ("Learning maps..")
		with open(data_file, 'r') as df:
			lines = df.readlines()
			for i in progressbar.progressbar(range(len(lines))):
				line = lines[i]
				try:
					split_index = line.rindex(',')
					question, qid = line[:split_index], int(line[split_index + 1: ])
					questions_map[qid] = question
				except:
					continue
				bigrams, trigrams = get_ngrams(question)		
				if bigrams:
					add_to_map(bigrams_map, bigrams, qid)
				if trigrams:
					add_to_map(trigrams_map, trigrams, qid)
			write_to_file(src_dir, BIGRAMS_SYNC, bigrams_map)
			write_to_file(src_dir, TRIGRAMS_SYNC, trigrams_map)
			write_to_file(src_dir, QUESTIONS_SYNC, questions_map)
	

def test(data_file):
	with open(data_file, 'r') as df:
		for line in df.readlines():
			t0 = time()

			#TODO remove before production, because question id will not be present. Using it to find duplicates in existing content
			try:
				split_index = line.rindex(',')
				question, qid = line[:split_index], int(line[split_index + 1: ])
			except:
				continue
			similar_question_ids = map(str, list(get_similar_questions(question)))

			#TODO uncomment
			#similar_questions = list(get_similar_questions(line))
			if similar_question_ids:
				similar_questions = [questions_map[id] for id in similar_question_ids]
				ranked_questions = rank_questions(line, similar_questions)
				print_vals(line, ranked_questions)
				#TODO remove
				"""
				if len(similar_question_ids) > 1:
					print ("Duplicates for", qid, question)
					for id in similar_question_ids:
						if id != qid:
							print ("Duplicate", id, questions_map[id])
					print ("----Duplicate---" * 4)
				"""
			else:
				print ("\n\nNo similar questions found for\n", line)
			print ("Time taken:", (time() - t0) * 1000, "ms")


if __name__=='__main__':
	from sys import argv
	t0_train = time()
	train(argv[1], argv[3])
	print ("Training time:", (time() - t0_train) * 1000, "ms")
	test(argv[2])
	
