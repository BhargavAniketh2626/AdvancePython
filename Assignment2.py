nm=[]
for x in range(2000, 3000):
    if (x%7==0) and (x%5!=0):
        nm.append(str(x))
print (','.join(nm))
