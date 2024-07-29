import csv 

training_data=[]

step =0
with open("ENJOYSPORT.csv",newline="")as csvfile:
    reader=csv.reader(csvfile)
    for row in reader:
        training_data.append(row)

    hypothesis=['?'] * (len(training_data[0]) -1)

    for example in training_data:
        if example[-1]=='1':
            for i in range (len(hypothesis)):
                if hypothesis[i]!=example[i]:
                    if hypothesis[i]=='?':
                        hypothesis[i]=example[i]
                    else:
                        hypothesis[i]='?'

        print("specific hypothesis :",hypothesis )
        step+=1
        print("\n")

print("final hypothesis : ",hypothesis)

