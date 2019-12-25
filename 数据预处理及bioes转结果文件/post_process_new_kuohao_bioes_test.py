import pandas as pd 
import re
import os

old_entities = []
train_df_r1 = pd.read_csv("r1_train_no_hand.csv", encoding="utf-8-sig")
train_df_r2 = pd.read_csv("r2_train_no_hand.csv", encoding="utf-8-sig")
train_df=pd.concat([train_df_r1,train_df_r2])

for x in list(train_df["unknownEntities"].fillna("")):
    old_entities.extend(x.split(";"))
old_entities = set(old_entities)
old_entities_df=list(old_entities)
old_entities_df=pd.DataFrame(old_entities_df)
old_entities_df.to_csv("remove_train.txt",index=None,header=None)
add_char = {']', '：', '~', '！', '%', '[', '《', '】', ';', '”', ':', '》', '？', '>', '/', '#', '。', '；', '&', '=', '，', '“', '【'}

def islegitimate(x):
    if re.findall("\\"+"|\\".join(add_char), x):
        return False
    if x in old_entities:
        return False

    return True
def check_brace(str_raw):
    # 如果传入为空，直接返回True
    if str_raw == "":
        return True

    # 定义一个空列表，模拟栈。
    stack = []

    while str_raw != "":
        # 获取本次循环的字符串的第一个字符
        thisChar = str_raw[0]
        # ^_^去掉第一个元素
        str_raw = str_raw[1:]
        # 如果本次循环的第一个字符是左括号，将其压栈
        if thisChar == "(" or thisChar == "（" or thisChar == "[":
            stack.append(thisChar)
        # 如果本次循环的第一个字符是右括号，检测栈是否为空，栈长为空表示栈内没有可以匹配的左括号，返回false
        # 如果栈长不为空，且栈内最后一个元素是相匹配的左括号，此次匹配成功，将其弹出，进入下一轮循环。
        elif thisChar == ")" or thisChar == "）" or thisChar == "]":
            # 提高效率
            len_stack = len(stack)
            if len_stack == 0:
                return False
            else:
                if thisChar == ")" and stack[len_stack - 1] == "(":
                    stack.pop(len_stack - 1)
                elif thisChar == "）" and stack[len_stack - 1] == "（":
                    stack.pop(len_stack - 1)
                elif thisChar == "}" and stack[len_stack - 1] == "{":
                    stack.pop(len_stack - 1)
                else:
                    return False
    # 循环结束，如果栈为空，则表示全部匹配完成
    if stack == []:
        return True
    else:
        return False

def extract_entity(res):

    sents = []
    word = ''
    part = []
    pos = []
    tag = ''
    tag_type = ''
    last_tag = ''
    last_tag_type = 'start'
    last_tag_pos = 'O'
    parts = []
    wbad_list = ['\u200b', '？', '《', '🔺', '️?', '!', '#', '%', '%', '，', 'Ⅲ', '》', '丨', '、', '​',
                 '👍', '。', '😎', '】', '⚠️', '：', '✅', '㊙️', '！', '🔥', ',', '“', '”', '；']
    for no, pre in enumerate(res):

        line = pre
        if line == '':
            if last_tag_type != 'o' \
                    and last_tag_pos == 'E' \
                    and 'B' in pos:
                parts.append(''.join(part).replace(',', ''))
            elif last_tag_type != 'o' \
                    and last_tag_pos == 'S':
                parts.append(''.join(part).replace(',', ''))

            parts=[i for i in parts if not i.isdigit() and len(i)>1]
            parts=[i for i in parts if [symbol for symbol in wbad_list if symbol in i]==[]]
            sents.append(parts)
            word = ''
            part = []
            tag = ''
            tag_type = ''
            last_tag_type = 'start'
            last_tag_pos = 'O'
            parts = []
        else:
            word, _, tag = line.split()

            if tag in ['O', 'X']:
                tag_type = 'o'
                tag_pos = 'O'
            else:
                tag+='-n'
                if '-' not in tag:
                    print(no, line)
                tag_pos, tag_type = tag.split('-')

            if last_tag_type == 'start':
                part.append(word)
                pos.append(tag_pos)
                last_tag_type = tag_type
                last_tag_pos = tag_pos
            elif tag_type == last_tag_type and tag_pos not in ['B', 'S']:
                part.append(word)
                pos.append(tag_pos)
                last_tag_type = tag_type
                last_tag_pos = tag_pos
            else:
                if last_tag_type != 'o' \
                        and last_tag_pos == 'E' \
                        and 'B' in pos:
                    parts.append(''.join(part).replace(',', ''))
                elif last_tag_type != 'o' \
                        and last_tag_pos == 'S':
                    parts.append(''.join(part).replace(',', ''))

                part = []
                pos = []
                part.append(word)
                pos.append(tag_pos)
                last_tag_type = tag_type
                last_tag_pos = tag_pos

    if len(part) > 0:
        if last_tag_type != 'o' \
                and last_tag_pos == 'E' \
                and 'B' in pos:
            parts.append(''.join(part).replace(',', ''))
        elif last_tag_type != 'o' \
                and last_tag_pos == 'S':
            parts.append(''.join(part).replace(',', ''))
        parts=[i for i in parts if not i.isdigit() and len(i)>1]
        parts=[i for i in parts if [symbol for symbol in wbad_list if symbol in i]==[]]
        sents.append(parts)
    return sents

def main():

    with open("result.txt", "r", encoding="utf-8-sig") as f:
        res = [line.strip() for line in f.readlines()]

    entity_list = extract_entity(res)

    new_entities = []

    for entities in entity_list:
        new = []
        for e in entities:
            if islegitimate(e):
                new.append(e)
        new_entities.append(new)
    
    if not os.path.exists("result"):
        os.mkdir("result")

    Dt_real = pd.read_csv("Round2_Test.csv", encoding="utf-8-sig")
    Dt = pd.read_csv("test_new.csv", encoding="utf-8-sig")
    Dt=Dt[~Dt['text'].isnull()]
    print(len(Dt))
    print(len(new_entities))
    assert len(Dt) == len(new_entities)
    Dt["unknownEntities"] = [";".join(set(x)).replace(",","") for x in new_entities]
    all_entity_dict=dict()
    for row in Dt.itertuples():
        all_entity_dict.setdefault(str(row.id),set())
        a_entity=row.unknownEntities.split(';')
        for i in a_entity:
            if i !="" and check_brace(i)==True:
                all_entity_dict[str(row.id)].add(i)
    train_new_df=[[i,";".join(all_entity_dict[i]).replace(",","")] for i in all_entity_dict]
    train_new_df=pd.DataFrame(train_new_df,columns=["id", "unknownEntities"])
    # train_new_df.columns=[["id", "unknownEntities"]]
    Dt_real[['id']]=Dt_real[['id']].astype(str)
    train_new_df[['id']]=train_new_df[['id']].astype(str)
    Dt_real=pd.merge(Dt_real[['id']],train_new_df[['id','unknownEntities']],on=['id'],how='left')
    print(Dt_real.head())
    Dt_real[["id", "unknownEntities"]].to_csv(f"./result/submit.csv", index=False, encoding="utf-8-sig")
    Dt[["id"]].to_csv(f"/home/ubuntu/zzp/bertNER/data_process/ensemble2/run_data/test_id_all_roberta.txt", index=False,header=None, encoding="utf-8-sig")
if __name__ == "__main__":
    main()
  
    pass
    
