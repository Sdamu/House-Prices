# kaggle 中的房价预测问题

### 读入数据
利用 pandas 读取 csv 文件中的数据

```{.python .input  n=10}
import pandas as pd
import numpy as np

train = pd.read_csv("E:\\AppData\\jupyter_proj\\House Prices\\data\\train.csv")
test = pd.read_csv("E:\\AppData\\jupyter_proj\\House Prices\\data\\test.csv")

all_X = pd.concat((train.loc[:, 'MSSubClass':'SaleCondition'],
                      test.loc[:, 'MSSubClass':'SaleCondition']))
```

```{.python .input  n=11}
train.head()
```

查看数据大小

```{.python .input  n=12}
print(train.shape)
print(test.shape)
```

### 对数据进行预处理
使用 pandas 对数值特征做白化处理

```{.python .input  n=14}
numeric_feats = all_X.dtypes[all_X.dtypes != "object"].index
all_X[numeric_feats] = all_X[numeric_feats].apply(lambda x: (x - x.mean())
                                                            / (x.std()))
```
