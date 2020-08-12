import glob
import re


# 레이블 제거한 텍스트와 info 파일[start,end,entity,word] 생성
labeled = re.compile('[<](.*?)[>](.*?)[<]/(.*?)[>]')
front = re.compile('[<](.*?)[>]')
back = re.compile('[<]/(.*?)[>]')
raw_compile = re.compile('[<]/?(.*?)[>]')  # 태그 제거하기 위한 정규표현식

# 계속 폴더명 바꿔줘야함..
filename = 'sent/2019_reviewed_deleted_3.txt.sent'

with open(filename,'r',encoding='utf-8') as r:
    text = r.read()

sep_name = filename[:-13]
info_name = sep_name + 'info.txt'

entity = re.finditer(labeled, text)
interval = 0

info = []
for i in entity:
    named = i.group()

    f = re.search(front, named)
    b = re.search(back, named)

    entity = named[1:f.end() - 1]
    word = named[f.end():b.start()]

    f_len = f.end() - f.start()
    b_len = b.end() - b.start()

    start = i.start() - interval
    interval += f_len + b_len
    end = i.end() - interval

    info.append([str(start),str(end),entity,word])

with open(info_name, 'w', encoding='utf-8') as f:
    for start, end, entity, word in info:
        f.write(start+'\t'+end+'\t'+entity+'\t'+word+'\n')

raw_name = sep_name + 'raw.txt.sent'
raw_text = re.sub(raw_compile, '', text)

with open(raw_name,'w',encoding='utf-8') as f:
    f.write(raw_text)

new_interval = 0
for start, end, entity, word in info:

    print(start,end,entity[:2],word)
    start = int(start)
    end = int(end)
    entity = entity[:2]
    if entity == 'PD':
        entity = 'DT'
    elif entity == 'GR':
        entity = 'OG'
    elif entity == 'PS':
        entity = 'PS'
    elif entity == 'LC':
        entity = 'LC'
    elif entity == 'OG':
        entity = 'OG'
    elif entity == 'TI':
        entity = 'TI'
    else:
        continue

    start += new_interval
    end += new_interval
    front = raw_text[:start]
    back = raw_text[end:]
    raw_text = front + "<" + word + ":" + entity + ">" + back
    new_interval += 5

new_name = sep_name + 'exo.txt.sent'

with open(new_name,'w',encoding='utf-8') as f:
    f.write(raw_text)


# 모든 문장을 정리한 파일 생성
with open('wisenut_11.txt','w',encoding='utf-8') as f:
    for filename in glob.iglob('different_result_3/**/*_intersection_exo.txt.sent', recursive=True):
        print(filename)
        with open(filename, 'r', encoding='utf-8') as r:
            text = r.read()

        f.write(text)








# 중간에 오류발생하는데 왜 나는지 모르겠음
# for filename in glob.iglob('different_result_3/**/*_intersection_text.txt.sent', recursive=True):
#     print(filename)
#     with open(filename,'r',encoding='utf-8') as r:
#         text = r.read()
#
#     sep_name = filename[:-13]
#     info_name = sep_name + 'info.txt'
#
#     entity = re.finditer(labeled, text)
#     interval = 0
#
#     info = []
#     for i in entity:
#         named = i.group()
#
#         f = re.search(front, named)
#         b = re.search(back, named)
#
#         entity = named[1:f.end() - 1]
#         word = named[f.end():b.start()]
#
#         f_len = f.end() - f.start()
#         b_len = b.end() - b.start()
#
#         start = i.start() - interval
#         interval += f_len + b_len
#         end = i.end() - interval
#
#         info.append([str(start),str(end),entity,word])
#
#     with open(info_name, 'w', encoding='utf-8') as f:
#         for start, end, entity, word in info:
#             f.write(start+'\t'+end+'\t'+entity+'\t'+word+'\n')
#
#     raw_name = sep_name + 'raw.txt.sent'
#     raw_text = re.sub(raw_compile, '', text)
#
#     with open(raw_name,'w',encoding='utf-8') as f:
#         f.write(raw_text)
#
#     new_interval = 0
#     for start, end, entity, word in info:
#
#         # print(start,end,entity[:2],word)
#         start = int(start)
#         end = int(end)
#         entity = entity[:2]
#         if entity == 'PD':
#             entity = 'DT'
#         elif entity == 'GR':
#             entity = 'OG'
#         elif entity == 'PS':
#             entity = 'PS'
#         elif entity == 'LC':
#             entity = 'LC'
#         elif entity == 'OG':
#             entity = 'OG'
#         elif entity == 'TI':
#             entity = 'TI'
#         else:
#             continue
#
#         start += new_interval
#         end += new_interval
#         front = raw_text[:start]
#         back = raw_text[end:]
#         raw_text = front + "<" + word + ":" + entity + ">" + back
#         new_interval += 5
#
#     new_name = sep_name + 'exo.txt.sent'
#
#     with open(new_name,'w',encoding='utf-8') as f:
#         f.write(raw_text)
#
#     with open(raw_name,'r',encoding='utf-8') as f:
#         new_text = f.read()
#
#     new_interval = 0
#
#     for start, end, entity, word in info:
#         start = int(start)
#         end = int(end)
#         type = entity[:2]
#         if type == 'PD':
#             type = 'DT'
#         elif type == 'GR':
#             type = 'OG'
#         elif type == 'PS':
#             type = 'PS'
#         elif type == 'LC':
#             type = 'LC'
#         elif type == 'OG':
#             type = 'OG'
#         elif type == 'TI':
#             type = 'TI'
#         else:
#             continue
#
#         start += new_interval
#         end += new_interval
#         front = new_text[:start]
#         end = new_text[end:]
#         new_text = front + "<" + word + ":" + type + ">" + end
#         new_interval += 5
#
#     new_name = sep_name + 'exo.txt.sent'
#
#     with open(new_name,'w',encoding='utf-8') as f:
#         f.write(new_text)


# with open('./data_in/wisenut_final.txt','w',encoding='utf-8') as f:
#     for filename in glob.iglob('different_result_3/**/*_intersection_exo.txt.sent', recursive=True):
#         print(filename)
#         with open(filename, 'r', encoding='utf-8') as r:
#             text = r.read()
#
#         f.write(text)




# with open('./data_in/wisenut_final.txt','w',encoding='utf-8') as f:
#     for filename in glob.iglob('different_result_3/**/*_intersection_raw.txt.sent', recursive=True):
#         with open(filename, 'r', encoding='utf-8') as r:
#             text = r.read()
#
#         sep_name = filename[:-12]
#         info_name = sep_name+'info.txt'
#         with open(info_name, 'r', encoding='utf-8') as r:
#             info_word = r.read()
#
#         info = []
#
#         info_word = info_word.split('\n')
#
#         for i in info_word:
#             info.append(i.split('\t'))
#         print(info)
#
#         new_interval = 0
#
#         for start, end, entity, word in info:
#
#             print(start,end,entity,word)
#             start = int(start)
#             end = int(end)
            # type = entity[:2]
            # if type == 'PD':
            #     type = 'DT'
            # elif type == 'GR':
            #     type = 'OG'
            # elif type == 'PS':
            #     type = 'PS'
            # elif type == 'LC':
            #     type = 'LC'
            # elif type == 'OG':
            #     type = 'OG'
            # elif type == 'TI':
            #     type = 'TI'
            # else:
            #     continue

        #     start += new_interval
        #     end += new_interval
        #     front = text[:start]
        #     back = text[end:]
        #     text = front + "<" + word + ":" + entity + ">" + end
        #     new_interval += (3+len(entity))
        #
        # f.write(text)











