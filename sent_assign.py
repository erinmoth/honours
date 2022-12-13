#Assigns sentiment score at sentence level
import pandas as pd
import sys
import csv
import re
import random
from sklearn.model_selection import train_test_split
from tensorflow import keras

#specified filenames
filename = sys.argv[1] + ".tsv"
compare = sys.argv[2] +".csv"
outname = sys.argv[3]+"_res.tsv"

df = pd.read_csv(filename, header=None, sep="\t", on_bad_lines='skip')   # reading into dataframe
tsv_file = df.values  
docs = pd.read_csv(compare, header=None, sep=",", on_bad_lines='skip')
subset = docs.sample(frac=1, random_state=7)

print("Analysing ", (subset.shape), "statements")
avgd={}
errors=0
with open(outname, 'wt') as out_file:
    tsv_writer = csv.writer(out_file, delimiter='\t')
    for sentence in subset.iloc[:,0]:
        #keeping track of information
        polarity=0
        pos = 0
        neg = 0
        count=0
        empty = 0
        words = str(sentence).split()
        for word in words:
            #clean data
            re.sub(r'[^\w\s]', '', word)     
            try:
                #find matches in lexicon
               relevant = df[df[0].str.fullmatch(word, case=False, na=False)] 
               if not relevant.empty:
                    val = relevant.iat[0,1]
                    polarity+=val
                    if(val>=0):
                        pos+=1
                    else:
                        neg+=1
                    count+=1
               else:
                    empty+=1
            except:
                errors +=1                

        if count != 0:
            avg = polarity/count
            avg2=(((pos-neg)+(avg*empty))/(count+empty))
            hard = 0
            if (avg2>0):
                hard = 1
            tsv_writer.writerow([sentence, hard])
print("encountered", errors, "errors")        

