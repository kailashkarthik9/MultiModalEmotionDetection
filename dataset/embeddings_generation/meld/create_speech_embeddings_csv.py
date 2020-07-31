import csv


def write(writer, file):
    for row in file:
        name_of_file = row.split(" ")[0]
        array = []
        for x in row.split(" ")[1:]:
            if x and x != '[' and x != ']\n':
                array.append(float(x.strip()))
        writer.writerow([name_of_file.split("-")[2], name_of_file.split("-")[3], name_of_file.split("-")[0],
                         name_of_file.split("-")[1], array])


output = open("meld/data/kaldi/meld_xvectors.csv", "w")
writer = csv.writer(output, delimiter=',')
writer.writerow(["dialogue_id", "utterance_id", "label", "set", "xvector"])

file1 = open("meld/data/kaldi/xvector.1.txt")
file2 = open("meld/data/kaldi/xvector.2.txt")
file3 = open("meld/data/kaldi/xvector.3.txt")
file4 = open("meld/data/kaldi/xvector.4.txt")
file5 = open("meld/data/kaldi/xvector.5.txt")

write(writer, file1)
write(writer, file2)
write(writer, file3)
write(writer, file4)
write(writer, file5)
