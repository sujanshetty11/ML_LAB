import csv 
traing_data=[]
step=0
with open("ENJOYSPORT.csv",newline="")as csvfile:
    reader=csv.reader(csvfile)
    for row in reader:
        traing_data.append(row)

g=['?']*(len(traing_data[0])-1)
s=[]

for example in traing_data:
    if example[-1]=='1':
        for i in range (len(g)):
            if g[i]!= example[i]:
                if g[i]=='?':
                   g[i]=example[i]
                else:
                    g[i]='?'
                for h in s[:]:
                    if any( h[j]!='?' and g[j]!=h[j]for j in range (len(g))):
                        s.remove(h)

    if example[-1]=='0':
        for i in range(len(g)):
            r=[]

            for z in range(0,6):
                if z==i:
                    r.append(g[i])
                else:
                    r.append('?')          

            for z in range (0,6):
                if r!='?':
                    s.append(r)
                    break

    print("STEP ",step)      
    print("general hypothesis :",g)
    print("specific hypothesis : ", s)
    step+=1
    print("\n")

print(" general hypothesis ", g)
print (" specific hypothesis :",s)