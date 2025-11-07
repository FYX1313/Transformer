# Transformer在IWSLT2017机器翻译中的应用

本代码手动实现了Transformer，并应用于IWSLT2017（zh-en）双语翻译数据集。
## 介绍
本文完整实现了Transformer的编码器-解码器框架，包括multi-head、self-attention、position-wise FFN、残差+LayerNorm、位置编码等。训练模型实现中文-英文的翻译任务。更改多头注意力机制的“头数”进行对比试验，并移除了位置编码部分进行了消融实验。



## 用法
1. 安装Pytorch和其他必要的依赖项。
```
pip install -r requirements.txt
```
2.训练和评估模型。我们在文件夹下提供实验脚本./scripts/。您可以通过下面命令复现实验结果：

```
bash ./scripts/run.sh
```
3.在进行消融实验的时候，请将Encoder.py以及Decoder.py文件中如下两行注释掉：
```
self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
x = self.pos_encoding(x.transpose(0, 1)).transpose(0, 1)
```
4.本实验未将数据集加载到本地然后再加载，而是直接在代码中联网获取数据集，所以请确保网络畅通。
```
dataset = load_dataset("iwslt2017", "iwslt2017-zh-en")
train_data = list(dataset['train'])
val_data = list(dataset['validation'])
test_data = list(dataset['test'])
```
## 实验结果
我们进行了15轮训练，得到了训练集、验证集各轮的损失，并采用最佳模型在测试集上进行测试，得到了测试集上的损失。为了更清楚地看到模型效果，我们提供了几个中文样本进行翻译测试。此外，我们还更改多头注意力的“头数”进行对比试验，并移除位置编码部分进行了消融实验。

### 训练、验证损失曲线及测试结果

<p align="center">
<img src=".\results\loss_plots\epoch_loss_curve_20251102_124329.png" width = "800" height = "" alt="" align=center />
</p>

### 不同“头数”的结果比较
下面的图像是我在生成各个头数的损失数据后另外绘制的。
<p align="center">
<img src=".\results\loss_plots\1.png" width = "800" height = "" alt="" align=center />
</p>
<p align="center">
<img src=".\results\loss_plots\2.png" width = "800" height = "" alt="" align=center />
</p>
<p align="center">
<img src=".\results\loss_plots\3.png" width = "800" height = "" alt="" align=center />
</p>

### 消融实验：移除位置编码
<p align="center">
<img src=".\results\loss_plots\epoch_loss_curve_20251106_195031.png" width = "800" height = "" alt="" align=center />
</p>

## GitHub链接
本代码以及放到GitHub上：
```
https://github.com/FYX1313/Transformer
```


