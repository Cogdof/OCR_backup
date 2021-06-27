line1 = input()


n = int(line1.split(" ")[0])
h = int(line1.split(" ")[1])

line2 = input()
line2 = line2.split(" ")
count=0

for i in range(0, n):
    if int(line2[i]) <= h:
        count+=1
    else:
        count+=2


print(count)