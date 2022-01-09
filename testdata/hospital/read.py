import csv
import random

dataframe = []
id1  = -1
list_temp = []
list_sources=['a','b','c','d','e']
with open('hospital_input_synthetic.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
	if id1 != row[0]:
		list_sources=['a','b','c','d','e']
		id1 = row[0]
	print row[0]
	source = random.choice(list_sources)
	row.append(source)
	list_sources.remove(source)

	dataframe.append(row)



import csv


csvfile = "hospital_fusion.csv"

#Assuming res is a flat list
with open(csvfile, "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    writer.writerows(dataframe)
