f = open("ng/nglist.txt", "w")

for i in range(1, 400) :
    s = str(i).zfill(6)
    # f.write(s + ".jpg 1 0 0 500 364\n")
    f.write("./ng/" + s + ".jpg\n")
f.close()
