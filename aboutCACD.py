# -*- coding:utf-8 -*-
from scipy.io import loadmat 
import numpy as np
cacd_path = '../DATA/CACD/celebrity2000_meta.mat'

m = loadmat(cacd_path)

# ### 输出所有的键：
# for key,item in m.items():
#     print(key)
# # celebrityImageData
# # celebrityData
# # __globals__
# # __header__
# # __version__
# ###----------------------------------

### 解析有关 celebrityImageData 的内容
# print(m['celebrityImageData'].shape)    #(1,1)

# print(np.dtype(m['celebrityData'][0][0]))
#[('age', 'O'), ('identity', 'O'), ('year', 'O'), ('feature', 'O'), ('rank', 'O'), ('lfw', 'O'), ('birth', 'O'), ('name', 'O')]

# for i in range(len(list(m['celebrityImageData'][0][0]))):
#     for j in range(len(list(m['celebrityImageData'][0][0])[0])):
#         print(list(m['celebrityImageData'][0][0])[i][j],end="  ")
#         ()

# a = m['celebrityImageData'][0][0]["identity"].tolist()
# print(a[20])
# for i in np.array(m['celebrityImageData'][0][0]):
#     print(i)
# print(m['celebrityImageData'][0][0])
###---------------------------------

# ### 解析有关 celebrityData 的内容
# print(m['celebrityData'])
# ###--------------------------------

def get_CACD(table_name,column_name,row_num,data_path="../DATA/CACD/celebrity2000_meta.mat"):
    """
    remember to
        <from scipy.io import loadmat>
        <import numpy as np>

    table_name: 
    |---celebrityImageData
    |---celebrityData

    column_name:
    |---celebrityImageData
        |---age
        |---identity
        |---year
        |---feature
        |---rank
        |---lfw
        |---birth
        |---name
    |---celebrityData
        |---name
        |---identity
        |---birth
        |---rank
        |---lfw
    """
    cacd_load = loadmat(data_path)
    list_already = cacd_load[table_name][0][0][column_name]
    return list_already[row_num]
if __name__ == "__main__":
    for i in range(20):
        print(get_CACD('celebrityImageData','name',i))
        print(get_CACD('celebrityImageData','age',i))