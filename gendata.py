
import csv
import matplotlib.pyplot as plt
import json


with open('data/ovpair.csv', 'r') as inputVOFile: 
    ovData = list(csv.reader(inputVOFile))


vo_dict = {}
ov_dict = {}
objects = []
for row in ovData:
    #print(row[0],0)
    obj = row[0].lower()
    verbs = []
    verbs_candidate = row[1:]
    for i in verbs_candidate:
        j = i.strip() #띄어쓰기 제거
        verbs.append(j)
        # all_verbs.append(j)


    ###ov pair###
    #if the verb-object pair is annotated to be valid
    if obj not in ov_dict: #obj가 없으면
        ov_dict[obj] = verbs #obj를 key로 가지는 list 생성
    if obj not in objects: #obj가 objects list에 없는 경우
        objects.append(obj) #list에 obj 추가


    ###vo pair###
    for v in verbs:
        if v not in vo_dict:
            vo_dict[v] = []
        if obj not in vo_dict[v]:
            vo_dict[v].append(obj)

# print(vo_dict) #710

###preprocessing verbs###
verbs = []
for v, obj in vo_dict.items():
    if (len(obj) >= 5): #verb에 해당하는 object가 5개 이상인 경우에
        verbs.append(v)

# print(len(verbs)) #165


##plot###
# x = []
# y = []
# i = 0
# for key in vo_dict.keys():
#     if len(vo_dict[key]) >= 5:
#         i += 1
#         x.append(i)
#         y.append(len(vo_dict[key]))
# print(i)
# plt.plot(x,y)
# plt.show()

###vo pair###
vo_dict2 = {}
ov_dict2 = {}

for verb in verbs:
    vo_dict2[verb] = vo_dict[verb]


objects = []

for verb, objs in vo_dict2.items():
    for obj in objs:
        if obj not in objects: #obj가 objects list에 없는 경우
            objects.append(obj) #list에 obj 추가


###ov pair###
for object in objects:
    for verb in ov_dict[object]:
        if object not in ov_dict2:
            ov_dict2[object] = []
        if verb in verbs:
            ov_dict2[object].append(verb)

# print(len(ov_dict2),11111) #594
# print(len(vo_dict2),22222) #165


##csv 저장###
with open('data/vopair.csv', 'w', newline='') as out: 
    csvwriter = csv.writer(out, delimiter=':')
    for verb in verbs:
        csvwriter.writerow((verb, vo_dict2[verb]))

with open('data/ovpair2.csv', 'w', newline='') as out: 
    csvwriter = csv.writer(out, delimiter=':')
    for object in objects:
        csvwriter.writerow((object, ov_dict2[object]))



###json 저장###
#ov_dict.json파일에 적절한 ov데이터만 저장
with open('data/ov_dict.json', 'w+') as f: 
    f.write(json.dumps(ov_dict2)) #json 문자열로 변환

with open('data/vo_dict.json', 'w+') as f: 
    f.write(json.dumps(vo_dict2)) #json 문자열로 변환



