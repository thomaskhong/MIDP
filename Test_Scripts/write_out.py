data = [60, 97, 107, 99, 58]

doc = open('C:/Users/khong/Documents/GitHub/MIDP/Test_Scripts/write_out.txt', 'w')

for i,x in enumerate(data):
    doc.write(str(x) + '\n')