import pandas as pd
import re
import os
import sys
import numpy as np
import string
from random import choice
sys.setrecursionlimit(1000) #例如这里设置为一百万
output_dir = "../SRC/data"

if not os.path.exists(output_dir):
    os.mkdir(output_dir)


def clean_str(input):

    input = input.replace(",", "，")
    input = input.replace("\xa0", "")
    input = input.replace("\b", "")
    input = input.replace('"', "")
    input = re.sub("\t|\n|\x0b|\x1c|\x1d|\x1e", "", input)
    input = input.strip()
    input = re.sub('\?\?+','',input)
    input = re.sub('\{IMG:.?.?.?\}','',input)
    input = re.sub('\t|\n','', input)
    # pattern1 = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')  # 剔除链接
    pattern1 = re.compile(
        r'((http|ftp|https)://)?(([a-zA-Z0-9\._-]+\.[a-zA-Z]{2,6})|([0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}))(:[0-9]{1,4})*(/[a-zA-Z0-9\&%_\./-~-]*)?')

    pattern2 = re.compile("\{IMG.*?\}")  # 剔除{IMG:1}{IMG:2}等等
    # pattern3 = re.compile("（.*?\）") # 剔除括号等等
    # pattern4 = re.compile("《.*?\》")  # 剔除括号等等
    # pattern5 = re.compile("【.*?】")  # 删除括号内容
    pattern6 = re.compile("\?{2,}")  # 删除多个问号
    pattern7 = re.compile(
        "[\w!#$%&'*+/=?^_`{|}~-]+(?:\.[\w!#$%&'*+/=?^_`{|}~-]+)*@(?:[\w](?:[\w-]*[\w])?\.)+[\w](?:[\w-]*[\w])?")  # 邮箱
    pattern8 = re.compile("0\d{2}-\d{8}|0\d{3}-\d{7}|\d{5}-\d{5}|\d{3}-\d{3}-\d{4}")  # 剔除电话
    pattern9 = re.compile("(20\d{2}([\.\-/|年月\s]{1,3}\d{1,2}){2}日?(\s?\d{2}:\d{2}(:\d{2})?)?)|(\d{1,2}\s?(分钟|小时|天)前)")  # 日期
    pattern10 = re.compile("<.*?>")
    punct = string.punctuation 
    pattern11 = re.compile("[^\u4e00-\u9fa5^a-z^A-Z^0-9%s]+" % punct)
    pattern12 = re.compile('^[?：！*]', re.S)
    pattern13 = re.compile("#")  # 删除#
    pattern14 = re.compile("\(\)|{}")  # 删除多个kuohao

    # shunxu
    pattern = [pattern11, pattern10, pattern2, pattern1, pattern6, pattern7, pattern8, pattern9, pattern14]
    # pattern = [pattern11, pattern10, pattern2, pattern1, pattern6, pattern7, pattern8, pattern14]
    pattern_2 = [pattern12, pattern13]


    # pattern_2 = [pattern4, pattern5, pattern12]
    def clean_zh(text):
        '''清洗文本，保证语句通顺(关于小数点的问题无法处理)'''
        # text = text.replace("（", "(").replace("）", ")")
        # punct = string.punctuation + punctuation
        # punct = "".join([c for c in punct if c not in [".", "、", "%", "“", "”", "(", ")", "！", "。", "？"]])
        # text = re.sub(r"[%s]+" % punct, " ", text)
        # 将引号替换
        # text = re.sub(r"[%s]+" % "“”()", "", text)
        # text = re.sub(r"[%s]+" % "：", " ", text)
        # 多个空格替换成一个
        # text = re.sub('/{2,}', '', text)
        # text = re.sub('\|{2,}', '', text)
        text = re.sub('window.public=.*\(window[,，]document\);', ' ', text)
        text = re.sub('varcontentConEle=.*AD_SURVEY_Add_AdPos\(\"42974\"\);', ' ', text)
        text = re.sub('&nbsp;|&quot;', '', text)
        text = re.sub('　+', ' ', text)
        text = re.sub(' +', ' ', text)
        text = re.sub('%+', '%', text)
        text = re.sub('#+', '#', text)

        # add ccp3
        text = re.sub("\[.*?\]", '', text)
        text = re.sub(",+", '，', text)

        #add
        text = text.replace("\xa0", "")
        text = text.replace("\b", "")
        text = re.sub("\t|\n|\x0b|\x1c|\x1d|\x1e", "", text)
        return text
    input=clean_zh(input)
    return input

def main():
    max_len=512
    max_len=max_len-2

    train_df = pd.read_csv("r2_train_hand.csv", encoding="utf-8-sig")

    test_df = pd.read_csv("r1_train_hand.csv", encoding="utf-8-sig")
    old_new_concat_df=pd.read_csv("r1_r2_concat_huibiao.csv", encoding="utf-8-sig")

    train_df['title']=train_df['title'].fillna('')
    test_df['title']=test_df['title'].fillna('')

    def juhao(x):
        if x=="":
            return x
        elif x[-1] not in '.。？！!':
            return x+'。'
        else:
            return x
        
    train_df[['title']]=train_df[['title']].applymap(lambda x:juhao(x))
    test_df[['title']]=test_df[['title']].applymap(lambda x:juhao(x))

    train_df['text'] =  train_df['title'].fillna('') + train_df['text'].fillna('')
    test_df['text'] =  test_df['title'].fillna('') + test_df['text'].fillna('')

    train_df['text'] = train_df['text'].apply(clean_str)
    test_df['text'] = test_df['text'].apply(clean_str)

    old_entities = []
    for x in list(old_new_concat_df["unknownEntities"].fillna("")):
        old_entities.extend(x.split(";"))
    old_entities = set(old_entities)
    old_entities_df=list(old_entities)
    old_entities_df.remove("")
    old_entities_df.sort(key=lambda x:len(x),reverse=True)
    train_df["unknownEntities"]=train_df["text"].apply(lambda x:';'.join([i for i in old_entities_df if i in x]))
    train_df["unknownEntities"]=train_df["unknownEntities"].apply(lambda x:np.nan if x=='' else x)

    train_no_entity = train_df[train_df['unknownEntities'].isnull()]
    print("没有实体的文本数量：",len(train_no_entity))
    train_df = train_df[~train_df['unknownEntities'].isnull()]#按位取反，就是把所有没实体的文本都去掉了。。。。。我感觉抽样一些出来比较好，而不是全去掉

    # 所有的非中文英文数字符号
    additional_chars = set()
    for t in list(test_df.text) + list(train_df.text):
        additional_chars.update(re.findall(u'[^\u4e00-\u9fa5a-zA-Z0-9\*]', t))
    
    # 一些需要保留的符号
    extra_chars = set("!#$%&\()*+,-./:;<=>?@[\\]^_`{|}~！#￥%&？《》{}“”，：‘’。（）·、；【】")
    additional_chars = additional_chars.difference(extra_chars)

    def remove_additional_chars(input):
        for x in additional_chars:
            input = input.replace(x, "")
        return input

    train_df["text"] = train_df["text"].apply(remove_additional_chars)
    test_df["text"] = test_df["text"].apply(remove_additional_chars)
    train_no_entity["text"] = train_no_entity["text"].apply(remove_additional_chars)

    def cut_sent(para):
        para = re.sub('([。！？\?])([^”’])', r"\1\n\2", para)  # 单字符断句符
        para = re.sub('(\.{6})([^”’])', r"\1\n\2", para)  # 英文省略号
        para = re.sub('(\…{2})([^”’])', r"\1\n\2", para)  # 中文省略号
        para = re.sub('([。！？\?][”’])([^，。！？\?])', r'\1\n\2', para)
        # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
        para = para.rstrip()  # 段尾如果有多余的\n就去掉它
        # 很多规则中会考虑分号;，但是这里我把它忽略不计，破折号、英文双引号等同样忽略，需要的再做些简单调整即可。
        return para.split("\n")

    def cut_to_min_len(sentence_list):
        sentence_list2=sentence_list.copy()
        for index,single_sentence in enumerate(sentence_list):
            if len(single_sentence)>max_len:
                last_symbol_index=-1
                for index2,i in enumerate(single_sentence):
                    if (i in '。！？?，！？\?;；’”】])）') and index2<=max_len-1:
                        last_symbol_index=index2+1
                    if index2>max_len:
                        break
                if last_symbol_index!=-1:
                    sentence_list2[index]=single_sentence[last_symbol_index:]
                    sentence_list2.insert(index,single_sentence[:last_symbol_index])
                    print(len(sentence_list2[index]))
                else:
                    sentence_list2[index]=single_sentence[max_len:]
                    sentence_list2.insert(index,single_sentence[:max_len])
                return cut_to_min_len(sentence_list2)

        return sentence_list


    def train_no_entity_df_process():
        not_real_num=0
        train_df_new=pd.DataFrame(columns=['id','text','unknownEntities'])
        for row in train_no_entity.itertuples():
            train_text_new_single=[]
            row_cut=cut_sent(row.text)

            row_cut=cut_to_min_len(row_cut)###在这里处理 分句后，有单句大于maxlen的句子的情况,处理成全部小于maxlen
            
            a_param=[]
            ################分句后所有句子都<=max_len
            for index,i in enumerate(row_cut):
                current_sen="".join(a_param)
                next_sen=''.join(a_param)+i
                if len(next_sen)>max_len:
                    train_text_new_single.append(current_sen)
                    if index==len(row_cut)-1:
                        train_text_new_single.append(i)
                    a_param=[i]
                else:
                    a_param.append(i)
                    if index==len(row_cut)-1:
                        current_sen="".join(a_param)
                        train_text_new_single.append(current_sen)
            flag=0
            for i in train_text_new_single:
                if len(i)<5:
                    continue
                train_df_new=train_df_new.append({'id':row.id,'text':i,'unknownEntities':np.nan},ignore_index=True)
                not_real_num+=1
        print("从没有实体的文本里采样得到：",not_real_num)
        return train_df_new

    def train_df_process():
        not_real_num=0
        max_not_real_num=2000
        train_df_new=pd.DataFrame(columns=['id','text','unknownEntities'])
        flag2=0
        for row in train_df.itertuples():
            train_text_new_single=[]
            row_cut=cut_sent(row.text)

            row_cut=cut_to_min_len(row_cut)###在这里处理 分句后，有单句大于maxlen的句子的情况,处理成全部小于maxlen
            
            a_param=[]
            ################分句后所有句子都<=max_len
            for index,i in enumerate(row_cut):
                current_sen="".join(a_param)
                next_sen=''.join(a_param)+i
                if len(next_sen)>max_len:
                    train_text_new_single.append(current_sen)
                    if index==len(row_cut)-1:
                        train_text_new_single.append(i)
                    a_param=[i]
                else:
                    a_param.append(i)
                    if index==len(row_cut)-1:
                        current_sen="".join(a_param)
                        train_text_new_single.append(current_sen)
            flag=[]
            for index,i in enumerate(train_text_new_single):
                if len(i)<5:
                    continue
                entitys = str(row.unknownEntities).split(';')
                for index2,j in enumerate(entitys):
                    if strQ2B(str(j)).lower() in strQ2B(str(i)).lower():  
                        train_df_new=train_df_new.append({'id':row.id,'text':i,'unknownEntities':row.unknownEntities},ignore_index=True)
                        break
                    if index2==len(entitys)-1:
                        flag.append(index)

            for i in flag  :
                if  not_real_num<=max_not_real_num :
                    train_df_new=train_df_new.append({'id':row.id,'text':train_text_new_single[i],'unknownEntities':np.nan},ignore_index=True)
                    not_real_num+=1
                # elif not_real_num>max_not_real_num and flag2==0:
                #     flag2=1
                #     print("负样本（从有实体的文本中采样）已到上限")
        print("负样本:",not_real_num)

        return train_df_new
    def test_df_process():
        test_df_new=pd.DataFrame(columns=['id','text'])
        for row in test_df.itertuples():
            train_text_new_single=[]
            row_cut=cut_sent(row.text)

            row_cut=cut_to_min_len(row_cut)###在这里处理 分句后，有单句大于maxlen的句子的情况,处理成全部小于maxlen
            
            a_param=[]
            ################分句后所有句子都<=max_len
            for index,i in enumerate(row_cut):
                current_sen="".join(a_param)
                next_sen=''.join(a_param)+i
                if len(next_sen)>max_len:
                    train_text_new_single.append(current_sen)
                    if index==len(row_cut)-1:
                        train_text_new_single.append(i)
                    a_param=[i]
                else:
                    a_param.append(i)
                    if index==len(row_cut)-1:
                        current_sen="".join(a_param)
                        train_text_new_single.append(current_sen)
            for i in train_text_new_single:
                if len(i)<5:
                    continue
                test_df_new=test_df_new.append({'id':row.id,'text':i},ignore_index=True)

        return test_df_new

    def strQ2B(ustring):
        """把字符串全角转半角"""
        ss = []
        for s in ustring:
            rstring = ""
            for uchar in s:
                inside_code = ord(uchar)
                if inside_code == 12288:  # 全角空格直接转换
                    inside_code = 32
                elif (inside_code >= 65281 and inside_code <= 65374):  # 全角字符（除空格）根据关系转化
                    inside_code -= 65248
                rstring += chr(inside_code)
            ss.append(rstring)
        return ''.join(ss)

    def corr_text(x):
        for i in error_text:
            if i in x:
                x=x.replace(i,error_text[i])
        return x
    error_ent=pd.read_csv("correct_entities.txt", encoding="utf-8-sig").values.tolist()
    error_ent={i[0]:i[1] for i in error_ent}
    error_text=pd.read_csv("correct_text.txt", encoding="utf-8-sig").values.tolist()
    error_text={i[0]:i[1] for i in error_text}

    train_df[['unknownEntities']]=train_df[['unknownEntities']].applymap(lambda x:error_ent[x] if x in error_ent else x)
    train_df[['text']]=train_df[['text']].applymap(lambda x:corr_text(x))
    test_df[['text']]=test_df[['text']].applymap(lambda x:corr_text(x))

    train_df=train_df_process()
    test_df=test_df_process()
    train_no_entity=train_no_entity_df_process()
    train_df=pd.concat([train_df,train_no_entity],axis=0)
    
    train_df.to_csv("train_new.csv",index=None, encoding="utf-8-sig")
    test_df.to_csv("test_new.csv",index=None, encoding="utf-8-sig")
    with open(f"{output_dir}/train_tag.txt", "w", encoding="utf-8-sig") as f:
        for row in train_df.itertuples():
            
            text_lbl = strQ2B(row.text.lower())
            entitys = strQ2B(str(row.unknownEntities)).lower().split(';')###############是否需要sort一下？？？？
            entitys.sort(key=lambda x:len(x),reverse=True)
            for entity in entitys:
                if len(entity)>1:
                    text_lbl = text_lbl.replace(entity, 'Ё' + (len(entity)-2)*'Ж'+'㉦')

            for c1, c2 in zip(row.text, text_lbl):
                if c2 == 'Ё':
                    f.write('{0}\t{1}\n'.format(c1, 'B'))
                elif c2 == 'Ж':
                    f.write('{0}\t{1}\n'.format(c1, 'I'))
                elif c2 == '㉦':
                    f.write('{0}\t{1}\n'.format(c1, 'E'))
                else:
                    f.write('{0}\t{1}\n'.format(c1, 'O'))
            
            f.write('\n')

    with open(f"{output_dir}/dev_tag.txt", "w", encoding="utf-8-sig") as f:
        for row in train_df.iloc[-200:].itertuples():
            
            text_lbl = strQ2B(row.text.lower())
            entitys = strQ2B(str(row.unknownEntities)).lower().split(';')###############是否需要sort一下？？？？
            entitys.sort(key=lambda x:len(x),reverse=True)
            for entity in entitys:
                if len(entity)>1:
                    text_lbl = text_lbl.replace(entity, 'Ё' + (len(entity)-2)*'Ж'+'㉦')

            for c1, c2 in zip(row.text, text_lbl):
                if c2 == 'Ё':
                    f.write('{0}\t{1}\n'.format(c1, 'B'))
                elif c2 == 'Ж':
                    f.write('{0}\t{1}\n'.format(c1, 'I'))
                elif c2 == '㉦':
                    f.write('{0}\t{1}\n'.format(c1, 'E'))
                else:
                    f.write('{0}\t{1}\n'.format(c1, 'O'))
            
            f.write('\n')

    with open(f"{output_dir}/test_tag_r1_train.txt", "w", encoding="utf-8-sig") as f:
        for row in test_df.itertuples():
            text_lbl = row.text
            for c1 in text_lbl:
                f.write('{0}\t{1}\n'.format(c1, 'O'))
            
            f.write('\n')

if __name__ == "__main__":
    main()
    pass
