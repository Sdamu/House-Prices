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

```{.json .output n=11}
[
 {
  "data": {
   "text/html": "<div>\n<style scoped>\n    .dataframe tbody tr th:only-of-type {\n        vertical-align: middle;\n    }\n\n    .dataframe tbody tr th {\n        vertical-align: top;\n    }\n\n    .dataframe thead th {\n        text-align: right;\n    }\n</style>\n<table border=\"1\" class=\"dataframe\">\n  <thead>\n    <tr style=\"text-align: right;\">\n      <th></th>\n      <th>Id</th>\n      <th>MSSubClass</th>\n      <th>MSZoning</th>\n      <th>LotFrontage</th>\n      <th>LotArea</th>\n      <th>Street</th>\n      <th>Alley</th>\n      <th>LotShape</th>\n      <th>LandContour</th>\n      <th>Utilities</th>\n      <th>...</th>\n      <th>PoolArea</th>\n      <th>PoolQC</th>\n      <th>Fence</th>\n      <th>MiscFeature</th>\n      <th>MiscVal</th>\n      <th>MoSold</th>\n      <th>YrSold</th>\n      <th>SaleType</th>\n      <th>SaleCondition</th>\n      <th>SalePrice</th>\n    </tr>\n  </thead>\n  <tbody>\n    <tr>\n      <th>0</th>\n      <td>1</td>\n      <td>60</td>\n      <td>RL</td>\n      <td>65.0</td>\n      <td>8450</td>\n      <td>Pave</td>\n      <td>NaN</td>\n      <td>Reg</td>\n      <td>Lvl</td>\n      <td>AllPub</td>\n      <td>...</td>\n      <td>0</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>0</td>\n      <td>2</td>\n      <td>2008</td>\n      <td>WD</td>\n      <td>Normal</td>\n      <td>208500</td>\n    </tr>\n    <tr>\n      <th>1</th>\n      <td>2</td>\n      <td>20</td>\n      <td>RL</td>\n      <td>80.0</td>\n      <td>9600</td>\n      <td>Pave</td>\n      <td>NaN</td>\n      <td>Reg</td>\n      <td>Lvl</td>\n      <td>AllPub</td>\n      <td>...</td>\n      <td>0</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>0</td>\n      <td>5</td>\n      <td>2007</td>\n      <td>WD</td>\n      <td>Normal</td>\n      <td>181500</td>\n    </tr>\n    <tr>\n      <th>2</th>\n      <td>3</td>\n      <td>60</td>\n      <td>RL</td>\n      <td>68.0</td>\n      <td>11250</td>\n      <td>Pave</td>\n      <td>NaN</td>\n      <td>IR1</td>\n      <td>Lvl</td>\n      <td>AllPub</td>\n      <td>...</td>\n      <td>0</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>0</td>\n      <td>9</td>\n      <td>2008</td>\n      <td>WD</td>\n      <td>Normal</td>\n      <td>223500</td>\n    </tr>\n    <tr>\n      <th>3</th>\n      <td>4</td>\n      <td>70</td>\n      <td>RL</td>\n      <td>60.0</td>\n      <td>9550</td>\n      <td>Pave</td>\n      <td>NaN</td>\n      <td>IR1</td>\n      <td>Lvl</td>\n      <td>AllPub</td>\n      <td>...</td>\n      <td>0</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>0</td>\n      <td>2</td>\n      <td>2006</td>\n      <td>WD</td>\n      <td>Abnorml</td>\n      <td>140000</td>\n    </tr>\n    <tr>\n      <th>4</th>\n      <td>5</td>\n      <td>60</td>\n      <td>RL</td>\n      <td>84.0</td>\n      <td>14260</td>\n      <td>Pave</td>\n      <td>NaN</td>\n      <td>IR1</td>\n      <td>Lvl</td>\n      <td>AllPub</td>\n      <td>...</td>\n      <td>0</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>0</td>\n      <td>12</td>\n      <td>2008</td>\n      <td>WD</td>\n      <td>Normal</td>\n      <td>250000</td>\n    </tr>\n  </tbody>\n</table>\n<p>5 rows \u00d7 81 columns</p>\n</div>",
   "text/plain": "   Id  MSSubClass MSZoning  LotFrontage  LotArea Street Alley LotShape  \\\n0   1          60       RL         65.0     8450   Pave   NaN      Reg   \n1   2          20       RL         80.0     9600   Pave   NaN      Reg   \n2   3          60       RL         68.0    11250   Pave   NaN      IR1   \n3   4          70       RL         60.0     9550   Pave   NaN      IR1   \n4   5          60       RL         84.0    14260   Pave   NaN      IR1   \n\n  LandContour Utilities    ...     PoolArea PoolQC Fence MiscFeature MiscVal  \\\n0         Lvl    AllPub    ...            0    NaN   NaN         NaN       0   \n1         Lvl    AllPub    ...            0    NaN   NaN         NaN       0   \n2         Lvl    AllPub    ...            0    NaN   NaN         NaN       0   \n3         Lvl    AllPub    ...            0    NaN   NaN         NaN       0   \n4         Lvl    AllPub    ...            0    NaN   NaN         NaN       0   \n\n  MoSold YrSold  SaleType  SaleCondition  SalePrice  \n0      2   2008        WD         Normal     208500  \n1      5   2007        WD         Normal     181500  \n2      9   2008        WD         Normal     223500  \n3      2   2006        WD        Abnorml     140000  \n4     12   2008        WD         Normal     250000  \n\n[5 rows x 81 columns]"
  },
  "execution_count": 11,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

查看数据大小

```{.python .input  n=12}
print(train.shape)
print(test.shape)
```

```{.json .output n=12}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "(1460, 81)\n(1459, 80)\n"
 }
]
```

### 对数据进行预处理
使用 pandas 对数值特征做白化处理

```{.python .input  n=14}
numeric_feats = all_X.dtypes[all_X.dtypes != "object"].index
all_X[numeric_feats] = all_X[numeric_feats].apply(lambda x: (x - x.mean())
                                                            / (x.std()))
```

```{.python .input}

```

```{.python .input}

```

```{.python .input}

```

```{.python .input}

```

```{.python .input}

```

```{.python .input}

```

```{.python .input}

```

```{.python .input}

```

```{.python .input}

```

```{.python .input}

```

```{.python .input}

```
