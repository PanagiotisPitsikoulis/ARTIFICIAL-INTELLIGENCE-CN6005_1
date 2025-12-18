fp = open("input.csv","r")

titles = []
values = []

cnt = 0

for line in fp:
    line = line.strip()
    print(line)

    words = line.split(", ")

    for word in words:
        print(word)
        if(cnt==0):
            titles.append(word)
        else:
            values.append(word)

    cnt += 1

for word in values:
    try:
        i = float(word)
        print(i, "is ok!!")
    except:
        print("provlhma me to ", word)
