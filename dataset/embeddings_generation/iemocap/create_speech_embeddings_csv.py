import csv


def write(writer, file, session):
    for row in file:
        name_of_file = row.split(" ")[0]
        array = []
        for x in row.split(" ")[1:]:
            # anger/disgust-test-01F-improvisation-01-007-F
            if x and x != '[' and x != ']\n':
                array.append(float(x.strip()))
        writer.writerow([name_of_file.split("-")[2], name_of_file.split("-")[3], name_of_file.split("-")[4],
                         name_of_file.split("-")[5], name_of_file.split("-")[6], name_of_file.split("-")[0], session,
                         array])


output = open("iemocap/data/kaldi/iemocap_xvectors.csv", "w")
writer = csv.writer(output, delimiter=',')
writer.writerow(
    ["session_id_mocap", "dialogue_type", "dialogue_id", "utterance_id", "gender", "label", "session_for_train",
     "xvector"])

file1 = open("iemocap/data/kaldi/xvector.1.txt")
file2 = open("iemocap/data/kaldi/xvector.2.txt")
file3 = open("iemocap/data/kaldi/xvector.3.txt")
file4 = open("iemocap/data/kaldi/xvector.4.txt")
file5 = open("iemocap/data/kaldi/xvector.5.txt")

write(writer, file1, 1)
write(writer, file2, 2)
write(writer, file3, 3)
write(writer, file4, 4)
write(writer, file5, 5)
