import os
import json

file_path = "wmt"
files = ['train', 'dev', 'test']
ch_path = 'corpus.ch'
en_path = 'corpus.en'
ch_lines = []
en_lines = []

for file in files:
    corpus = json.load(open(os.path.join(file_path, file + '.json'), 'r'))
    for item in corpus:
        ch_lines.append(item[1] + '\n')
        en_lines.append(item[0] + '\n')

with open(os.path.join(file_path, ch_path), "w") as fch:
    fch.writelines(ch_lines)

with open(os.path.join(file_path, en_path), "w") as fen:
    fen.writelines(en_lines)

# lines of Chinese: 252777
print("lines of Chinese: ", len(ch_lines))
# lines of English: 252777
print("lines of English: ", len(en_lines))
print("-------- Get Corpus ! --------")