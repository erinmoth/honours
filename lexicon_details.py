#Collects information about lexions - size and word categorisation 
#imports
import pandas as pd
import sys
import csv
import re
from sklearn.model_selection import train_test_split
import string

#open specified files
lexicon = sys.argv[1] + ".tsv"
scraped = sys.argv[2] +".csv"
out = sys.argv[3]
df = pd.read_csv(lexicon, header=None, sep="\t")  
lexicon_set = set(df.iloc[:,0])

docs = pd.read_csv(scraped, header=None, sep=",", on_bad_lines='skip')
subset = docs.sample(frac=1, random_state=7)
sentence_size = 0
num_sentences = 0
no_sent = 0
no_match = []
wordlist = []

for sentence in subset.iloc[:,0]:
    num_sentences+=1
    words = str(sentence).split()
    #counts duplicates
    for word in words:
        sentence_size+=1
        word = str(word).lower()
        word = word.translate(str.maketrans('', '', string.punctuation))
        re.sub(r'[^\w\s]', '', word)
        if word not in lexicon_set:
            no_match.append(word)
            no_sent += 1
    

#print summary
print(str(num_sentences) + " sentences")
print(str(no_sent) + " no sentiment words")
print(str(len(set(no_match))) + " unique no-match words")
wps = sentence_size / num_sentences
print(str(wps) + " words per sentence")
nsps = no_sent / num_sentences
print(str(nsps) + " no sentiment per sentence")
outfile = out + "_no_sent_words.csv"
print("Unique sent words saved to " + outfile)

#save words with no sentiment for manual review
pd.DataFrame(set(no_match)).to_csv(outfile, index = False)

