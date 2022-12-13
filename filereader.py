import sys
import csv
#Read in scraped data for each domain and combine for labelling
print('Reading in:', (len(sys.argv)-2), 'files.')
files = sys.argv
outname = files[1] +"_domain.tsv"
files = files[2:]
print('Files to be read:', files)
averages= {}
count = {}
for argument in files:
    filename = argument + ".tsv"
    with open(filename) as file:
        tsv_file = csv.reader(file, delimiter="\t")
        print("Reading ", filename)
        for line in tsv_file:
            name = line[0]
            score = line[1]
            if name in averages:
               # print("Updating ", name, " with score ", averages[name])
                averages[name] += float(score)
                count[name] += 1
               # print("New value: ", averages[name], "with ", count[name], " entries")
            else:
                averages[name] = float(score)
                count[name] = 1

with open(outname, 'wt') as out_file:
    tsv_writer = csv.writer(out_file, delimiter='\t')
    for name in averages:
        avg_score = (float(averages[name])/float(count[name]))
        tsv_writer.writerow([name, avg_score])
  

                
