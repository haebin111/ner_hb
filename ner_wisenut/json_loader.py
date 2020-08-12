import json

def loader(json_path):

    j = json.load(open(json_path, 'r', encoding='utf-8'))

    data = j['data']
    items = j['attributes']['entities']['items']



    ner_list = []
    for i in items:
        ner = i['type']
        len_ner = len(ner)
        start = i['mentions'][0]['startOffset']
        end = i['mentions'][0]['endOffset']
        word = data[start:end]
        word_ls = word.lstrip()
        word_rs = word.rstrip()
        if len(word) != len(word_ls):
            word = word_ls
            start += 1
        elif len(word) != len(word_rs):
            word = word_rs
            end -= 1
        ner_list.append([word, ner, start, end])

    ner_list.sort(key=lambda x: x[2])


    text = data
    length = 0
    for word, ner, start, end in ner_list:
        new_start = start + length
        new_end = end + length
        front = text[:new_start]
        back = text[new_end:]
        text = front + "<" + word + ":" + ner + ">" + back
        length += (len(ner) + 3)



    data = data.replace("\r", "").replace("\u3000", "")
    pure = data.split("\n")
    with open('data/pure/2015_reviewed_deleted_17_pure.txt', 'w', encoding='utf-8') as f:
         for line in pure:
             f.write(line + "\n")
         f.close()

    text = text.replace("\r", "").replace("\u3000", "")
    labeled = text.split("\n")
    with open('data/labeled/2015_reviewed_deleted_17_labeled.txt', 'w', encoding='utf-8') as f:
         for line in labeled:
             f.write(line + "\n")
         f.close()

    return pure, labeled

pure, labeled = loader('')
print(pure[0])
print(labeled[0])
