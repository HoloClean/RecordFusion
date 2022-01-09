import csv
import pandas as pd

flaten = []
with open('input_golden_record.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
	    if line_count ==0:
		attributes = row
		line_count = line_count + 1
	    else:
		for i in range(1, len(row)):
			flaten.append([row[0], attributes[i], str(row[i])])
df = pd.DataFrame(flaten, columns=["tid" , "attribute" ,"correct_val"])
df.to_csv("flaten_input_record.csv", sep=',', encoding='utf-8',  index=False)
