import numpy as np
import math
import pandas as pd

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
# test = pd.read_csv('train.csv')
answer = pd.read_csv('answers.csv')

NofSpam = 0
NofHam = 0
trainColumns = train.columns
print(len(trainColumns))

nIfSpamOfEach = {}
NotnIfSpamOfEach = {}

nIfHamOfEach = {}
NotnIfHamOfEach = {}

prIfSpamOfEach = {}
NotprIfSpamOfEach = {}

prIfHamOfEach = {}
NotprIfHamOfEach = {}

print(len(train.index))
# pr of spam and ham

for i in range(0, len(train.index)):
    if train["Spam"][i] == 1:
        NofSpam += 1
    else:  # is Ham
        NofHam += 1

prOfSpam = NofSpam * (1.0) / (NofHam + NofSpam)
prOfHam = NofHam * (1.0) / (NofHam + NofSpam)
print(prOfSpam, prOfHam)

# learn
for j in range(0, len(trainColumns) - 1):  # for each word
    word = trainColumns[j]
    nIfSpamOfEach[word] = 0
    NotnIfSpamOfEach[word] = 0
    nIfHamOfEach[word] = 0
    NotnIfHamOfEach[word] = 0

    for i in range(0, len(train.index)):  # looking in every example
        if train["Spam"][i] == 1:

            if train[word][i] == 1:
                nIfSpamOfEach[word] = nIfSpamOfEach[word] + 1
            else:
                NotnIfSpamOfEach[word] = NotnIfSpamOfEach[word] + 1
        else:  # is Ham

            if train[word][i] == 1:
                nIfHamOfEach[word] = nIfHamOfEach[word] + 1
            else:
                NotnIfHamOfEach[word] = NotnIfHamOfEach[word] + 1
    # settin probs
    prIfSpamOfEach[word] = nIfSpamOfEach[word] * (1.0) / NofSpam
    NotprIfSpamOfEach[word] = NotnIfSpamOfEach[word] * (1.0) / NofSpam

    prIfHamOfEach[word] = nIfHamOfEach[word] * (1.0) / NofHam
    NotprIfHamOfEach[word] = NotnIfHamOfEach[word] * (1.0) / NofHam

    print(word, prIfSpamOfEach[word], NotprIfSpamOfEach[word], prIfHamOfEach[word], NotprIfHamOfEach[word])

# predict
testColumns = test.columns
perdictions = []
print("lenght of test is :" + str(len(test.index)))
error = 0
base = 10

for i in range(1, len(test.index)):  # each test example
    res = math.log(prOfSpam / prOfHam, base)
    for j in range(0, len(testColumns) - 1):  # for each word
        word = trainColumns[j]  # the word
        # print(word)
        if test[word][i] == 1:
            res += math.log(prIfSpamOfEach[word], base)
            res -= math.log(prIfHamOfEach[word], base)
        else:
            res += math.log(NotprIfSpamOfEach[word], base)
            res -= math.log(NotprIfHamOfEach[word], base)

    if res >= 0:
        perdictions.append(1)
    else:
        perdictions.append(0)
    if perdictions[i - 1] != answer.iloc[i - 1][0]:
        # if perdictions[i] != train["Spam"][i]:
        error += 1
        print(
            str(error) + "th mismatch! in row: -> " + str(i) + " <- predicted: " + str(
                perdictions[i - 1]) + " answer is: " + str(
                answer.iloc[i][0]) + " ==> result: " + str(res))
        # train["Spam"][i]) + "result: " + str(res))
    else:
        print(str(i) + "th is ok answer was :" + str(perdictions[i - 1]) + " ==> result: " + str(res))
output = pd.DataFrame(perdictions)
output.to_csv("SPM_9531057.csv", index=False)
print("end")
