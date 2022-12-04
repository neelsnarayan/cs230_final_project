import random

TRAIN_PERCENT = 0.8
DEV_PERCENT = 0.1
TEST_PERCENT = 0.1

assert (
    TRAIN_PERCENT + DEV_PERCENT + TEST_PERCENT == 1.0
), "The percentages do not total 100!"

with open("flickr_audio/wav2spk.txt", "r") as f:
    lines = f.read().split("\n")
del lines[-1]

random.shuffle(lines)

train = lines[: int(len(lines) * TRAIN_PERCENT)]
dev = lines[len(train) : int(len(lines) * DEV_PERCENT) + len(train)]
test = lines[len(train) + len(dev) :]

assert len(train) + len(dev) + len(test) == len(
    lines
), "The distributions are not created properly!"

with open("wav2spk_TRAIN.txt", mode="w", encoding="utf-8") as f:
    f.write("\n".join(train))

with open("wav2spk_DEV.txt", mode="w", encoding="utf-8") as f:
    f.write("\n".join(dev))

with open("wav2spk_TEST.txt", mode="w", encoding="utf-8") as f:
    f.write("\n".join(test))
