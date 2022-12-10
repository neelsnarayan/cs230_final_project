import glob, os

FILE_PATH = "train_audio2/JL(wav+txt)/"
os.chdir(FILE_PATH)

files = []
speakers = []
for file in glob.glob("*.wav"):
    files.append(file)
    speakers.append(file.split("_")[0])

speaker_mappings = {"male1": 0, "male2": 1, "female1": 2, "female2": 3}
speaker_indices = [speaker_mappings[s] for s in speakers]

return_string = ""
i = 0
for f in files:
    return_string += f + " " + str(speaker_indices[i]) + "\n"
    i += 1

text_file = open("wav2spk_test.txt", "w")
text_file.write(return_string)
text_file.close()
