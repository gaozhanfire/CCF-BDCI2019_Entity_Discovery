import pandas as pd
import numpy as np
import os

all_test_num=len([i for i in os.listdir('all_result/') if '.csv' in i ])
all_csv_file=[i for i in os.listdir('all_result') if '.csv' in i ]
all_ent=[]
flag=0
for index,single_csv in enumerate(os.listdir('all_result')):
    if '.csv' in single_csv and 'dev' not in single_csv:
        if os.path.getsize("all_result/"+single_csv)<16384000:
            single_csv=pd.read_csv("all_result/"+single_csv)
            if flag==0:
                flag=1
                all_ent=[[] for i in range(len(single_csv))]
            for row_index,row in enumerate(single_csv['unknownEntities']):
                try:
                    row_ent=row.split(';')
                    all_ent[row_index].extend(row_ent)
                except:
                    continue
# all_ent
zuizhong_ent=[]
for row_ent in all_ent:
    row_process_ent=dict()
    for i in row_ent:
        row_process_ent.setdefault(i,0)
        row_process_ent[i]+=1
    toupiao_result=[]
    for i in row_process_ent:
        if row_process_ent[i]>=0:
            toupiao_result.append(i)
    zuizhong_ent.append(toupiao_result)

zuizhong_ent=[';'.join(i) if i!=[] else np.nan for i in zuizhong_ent]
id_col=pd.read_csv("all_result/"+all_csv_file[0])
all_result=pd.DataFrame(zuizhong_ent,columns=['unknownEntities'])
all_result_df=pd.concat([id_col['id'],all_result],axis=1)
all_result_df.to_csv("es_result/bingji.csv",index=None)
