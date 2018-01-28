#!/bin/bash

if [ $1 -eq 1 ]
	then
	echo python code/hpf.py -d=data/train/text8/text8_top_10000_w_2_data.npz -o=word_vec/ -k=10 -v -n=101 -i=trial
	python code/hpf.py -d=data/train/text8/text8_top_10000_w_2_data.npz -o=word_vec/ -k=10 -v -n=101 -i=trial
elif [ $1 -eq 2 ]
	then
	echo python code/bpf.py -d=data/sotu/sotu_top_2000_w_2_data.npz -o=word_vec/ -k=10 -v -n=101 -i=trial
	python code/bpf.py -d=data/sotu/sotu_top_2000_w_2_data.npz -o=word_vec/ -k=10 -v -n=101 -i=trial
fi