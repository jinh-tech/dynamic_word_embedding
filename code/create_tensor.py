import numpy as np
from os import listdir

prefix_data = "data/"
dataset = "sotu"
dataset_files = "sotu_files/"

all_files = [f for f in listdir(prefix_data + dataset + '/' + dataset_files) ]
max_sentence_size = 4
context_window = [2]  # i.e x words to the right and x words to the left
top_words = [2000]

word_to_index = {}
no_words = 0
index_to_freq = {}

for i in range(0,len(all_files)):

	with open(prefix_data+ dataset+'/'+dataset_files + all_files[i],'r') as f:
		data = f.read().lower().split('.')
		count = 0
		temp_index_count = {}
		for line in data:
			for j in line.split():
				if not (j in word_to_index):
					word_to_index[j] = no_words
					no_words += 1

				if word_to_index[j] in temp_index_count:
					temp_index_count[word_to_index[j]] += 1
				else:
					temp_index_count[word_to_index[j]] = 1
				count += 1	

		for ind,c in temp_index_count.iteritems():
			if ind in index_to_freq:
				index_to_freq[ind] += c
			else:
				index_to_freq[ind] = c

frq_word = np.zeros(no_words,dtype=np.float)

for ind,c in index_to_freq.iteritems():
	frq_word[ind] = c

ind_frq_word = np.argsort(frq_word)[::-1]
index_to_word = {v: k for k, v in word_to_index.iteritems()}

print "Number of words are %d"%no_words

for window_size in context_window:
	for top in top_words:

		rand_ind = np.arange(0,top)
		np.random.shuffle(rand_ind)
		c = 0
		temp_word_to_ind = {}
		for ind in xrange(0,top):
			temp_word_to_ind[index_to_word[ind_frq_word[ind]]] = rand_ind[c]
			c += 1
		subs = [[],[],[]]
		vals = []
		for i in range(0,len(all_files)):
			with open(prefix_data+dataset+'/'+dataset_files + all_files[i],'r') as f:
				indices = {}
				data = f.read().lower().split('.')
				for j in range(0,len(data)):
					line = data[j].split()
					if len(line) > max_sentence_size:
							for k in xrange(0,len(line)-(window_size+1)):
								if line[k] in temp_word_to_ind:
									for l in range(1,window_size+1):
										if line[k+l] in temp_word_to_ind:
											temp_ind = (word_to_index[line[k]],word_to_index[line[k+l]])
											if temp_ind in indices:
												indices[temp_ind] += 1
											else:
												indices[temp_ind] = 1
											temp_ind = (word_to_index[line[k+l]],word_to_index[line[k]])
											if temp_ind in indices:
												indices[temp_ind] += 1
											else:
												indices[temp_ind] = 1

			for ind,val in indices.iteritems():
				subs[0].append(temp_word_to_ind[index_to_word[ind[0]]])
				subs[1].append(temp_word_to_ind[index_to_word[ind[1]]])
				subs[2].append(i)
				vals.append(val)
		with open(prefix_data+dataset+"/"+dataset+"_top_"+str(top)+"_w_"+str(window_size)+"_data.npz", 'w') as f:
			np.savez(f, indices=subs, vals=vals, size=(top,top,len(all_files)))

		with open(prefix_data+dataset+"/"+dataset+"_top_"+str(top)+"_w_"+str(window_size)+"_dict.txt", 'w') as f:
			for ind,val in temp_word_to_ind.iteritems():
				f.write(str(ind)+" "+str(val)+"\n")
		
		print "Finished for no_words = %d"%top
