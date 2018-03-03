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
print(len(numeric_feats))
print(numeric_feats)
all_X[numeric_feats] = all_X[numeric_feats].apply(lambda x: (x - x.mean())
                                                            / (x.std()))
```

将离散数据点转换成数值标签

```{.python .input}
all_X = pd.get_dummies(all_X,dummy_na=True)
all_X
```

将缺失数据用本特征的平均值进行估计

```{.python .input}
all_X = all_X.fillna(all_X.mean())
```

将数据转换一下格式

```{.python .input}
num_train = train.shape[0]

X_train = all_X[:num_train].as_matrix()
X_test = all_X[num_train:].as_matrix()
y_train = train.SalePrice.as_matrix()
```

## 导入 NDArray 格式数据
为了方便和 Gluon 交互，我们需要导入 NDArray 格式的数据

```{.python .input}
from mxnet import gluon
from mxnet import ndarray as nd
from mxnet import autograd

X_train = nd.array(X_train)
y_train = nd.array(y_train)
y_train.reshape((num_train,1))

X_test = nd.array(X_test)
```

将损失函数定义为平方损失

```{.python .input}
square_loss = gluon.loss.L2Loss()
```

定义比赛中测量结果用的函数。

```{.python .input}
def get_rmse_log(net, X_train, y_train):
    num_train = X_train.shape[0]
    clipped_preds = nd.clip(net(X_train), 1, float('inf'))
    return np.sqrt(2 * nd.sum(square_loss(
        nd.log(clipped_preds),nd.log(y_train))).asscalar() / num_train)
```

## 定义模型
我们将模型的定义放在一个函数里供多次调用。这是一个基本的线性回归模型。

```{.python .input}
def get_net():
    net = gluon.nn.Sequential()
    with net.name_scope():
        net.add(gluon.nn.Dense(32))
        net.add(gluon.nn.Dense(1))
    net.initialize()
    return net
```

我们定义一个训练的函数，这样在跑不同的实验时不需要重复实现相同的步骤。

```{.python .input}
%matplotlib inline
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 120
import matplotlib.pyplot as plt

def train(net, X_train, y_train, X_test, y_test, epochs,
          verbose_epoch, learning_rate, weight_decay):
    train_loss = []
    if X_test is not None:
        test_loss = []
    batch_size = 100
    dataset_train = gluon.data.ArrayDataset(X_train, y_train)
    data_iter_train = gluon.data.DataLoader(
        dataset_train, batch_size, shuffle=True)
    trainer = gluon.Trainer(net.collect_params(),'adam',
                           {'learning_rate': learning_rate,
                           'wd': weight_decay})
    net.collect_params().initialize(force_reinit=True)
    for epoch in range(epochs):
        for data,label in data_iter_train:
            with autograd.record():
                output = net(data)
                loss = square_loss(output, label)
            loss.backward()
            trainer.step(batch_size)
            
            cur_train_loss = get_rmse_log(net, X_train, y_train)
        if epoch > verbose_epoch:
            print("Epoch %d, train loss: %f" %(epoch, cur_train_loss))
        train_loss.append(cur_train_loss)
        if X_test is not None:
            cur_test_loss = get_rmse_log(net, X_test, y_test)
            test_loss.append(cur_test_loss)
    plt.plot(train_loss)
    plt.legend(['train'])
    if X_test is not None:
        plt.plot(test_loss)
        plt.legend(['train', 'test'])
    plt.show
    if X_test is not None:
        return cur_train_loss, cur_test_loss
    else:
        return cur_train_loss


```

## K折交叉验证
在过拟合中提到过，过度依赖训练数据集的误差来推断测试数据集的误差容易导致过拟合的出现。事实上，当我们在调参的时候，往往需要基于 K 折交叉验证。

> 在 K 折交叉验证中，我们把初始采样分割成为 K 个子样本，一个单独的子样本被保留作为验证模型的数据，其他 K-1 个样本用来进行训练。

我们关心 K 次验证模型的测试结果的平均值和训练误差的平均值，因为我们定义 K 折交叉验证函数如下所示：

```{.python .input}
def k_fold_cross_valid(k, epochs, verbose_epoch, X_train, y_train,
                       learning_rate, weight_decay):
    assert k > 1
    fold_size = X_train.shape[0] // k
    train_loss_sum = 0.0
    test_loss_sum = 0.0
    for test_i in range(k):
        X_val_test = X_train[test_i * fold_size: (test_i + 1) * fold_size, :]
        y_val_test = y_train[test_i * fold_size: (test_i + 1) * fold_size]

        val_train_defined = False
        for i in range(k):
            if i != test_i:
                X_cur_fold = X_train[i * fold_size: (i + 1) * fold_size, :]
                y_cur_fold = y_train[i * fold_size: (i + 1) * fold_size]
                if not val_train_defined:
                    X_val_train = X_cur_fold
                    y_val_train = y_cur_fold
                    val_train_defined = True
                else:
                    X_val_train = nd.concat(X_val_train, X_cur_fold, dim=0)
                    y_val_train = nd.concat(y_val_train, y_cur_fold, dim=0)
        net = get_net()
        train_loss, test_loss = train(
            net, X_val_train, y_val_train, X_val_test, y_val_test, 
            epochs, verbose_epoch, learning_rate, weight_decay)
        train_loss_sum += train_loss
        print("Test loss: %f" % test_loss)
        test_loss_sum += test_loss
    return train_loss_sum / k, test_loss_sum / k
```

## 训练模型并进行交叉验证
以下的模型参数都是可以调节的。

```{.python .input}
k = 7
epochs = 100
verbose_epoch =0
learning_rate = 0.05
weight_decay = 5.0
```

给定以上调好的参数，接下来我们训练并交叉验证我们的模型。

```{.python .input}
train_loss, test_loss = k_fold_cross_valid(k, epochs, verbose_epoch, X_train,
                                        y_train, learning_rate, weight_decay)
print("%d-fold validation: Avg train loss: %f, Avg test loss: %f" %
     (k, train_loss, test_loss))
```

即便训练误差可以达到很低（调好参数之后），但是K折交叉验证上的误差可能更高。当训练误差特别低时，要观察K折交叉验证上的误差是否同时降低并小心过拟合。我们通常依赖K折交叉验证误差结果来调节参数。



## 预测并在Kaggle提交预测结果（选学）

本部分为选学内容。网络不好的同学可以通过上述K折交叉验证的方法来评测自己训练的模型。

我们首先定义预测函数。

```{.python .input}
def learn(epochs, verbose_epoch, X_train, y_train, test, learning_rate,
          weight_decay):
    net = get_net()
    train(net, X_train, y_train, None, None, epochs, verbose_epoch, 
          learning_rate, weight_decay)
    preds = net(X_test).asnumpy()
    test['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])
    submission = pd.concat([test['Id'], test['SalePrice']], axis=1)
    submission.to_csv('submission.csv', index=False)
```

调好参数以后，下面我们预测并在Kaggle提交预测结果。

```{.python .input}
learn(epochs, verbose_epoch, X_train, y_train, test, learning_rate,
      weight_decay)
```

```{.python .input}

```
