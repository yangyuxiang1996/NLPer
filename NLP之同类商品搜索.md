# NLP之同类商品搜索

## 项目背景

当用户在网上购买商品时经常会试着货比三家，比如某一个京东的商品在苏宁网上的价格是怎样的。 为了便于这种比较，京东开发了一个同类商品搜索模块：给定一个京东商品，它可以根据商品相关的信息去自动找到苏宁等平台上的同类商品。 这里的一个难点在于，每一个商品在不同平台上的标题、描述这些都有一些区别的，所以定位到同一个商品本身具有一定的挑战。

<img src="/Volumes/yyx/学习/NLP知识体系/images/image-20210706220935920.png" alt="image-20210706220935920" style="zoom:50%;" />

<img src="/Volumes/yyx/学习/NLP知识体系/images/image-20210706211002113.png" alt="image-20210706211002113" style="zoom:50%;" />

## 项目数据

整个项目用到了京东和苏宁电商平台上的产品的SKU，包括：

* 电商产品的相关属性；
* 产品词表

数据量大约5万多个产品SKU，以JsonLine的方式储存：

* kb_jd_jsonl.txt 为京东sku数据，每行为一个sku的描述信息，jsonl格式，其中base_main_sku_id字段为商品唯一id，base_item_url为商品链接，其余为商品属性；
* kb_sn_jsonl.txt 为苏宁sku数据，格式与京东sku数据相同；
* train.txt, eval.txt, test.txt 分别为训练集、验证集、测试集，格式均为“京东sku_id	苏宁sku_id	是否匹配(1/0)”，中间以tab分割。

数据示例：

```json
{
    "操作系统": "64位及以上或android5.0及以上",
    "支持音频格式": "MP3，AAC,WMA",
    "支持视频格式": "H.265：视频解码，MP4，3GP，AVI，MPG，RM，RMVB，MOV，MKV，MPEG",
    "4K显示": "不支持",
    "动态补偿": "不支持",
    "型号": "LED32Y3A",
    "内存": "1GB+8GB",
    "CPU": "4核",
    "边框宽窄": "窄边款",
    "背光方式": "直下式",
    "数字视频接口": "HDMI 2.0a接口",
    "HDMI1.3接口数": "无",
    "视频牌照商": "iCNTV腾讯",
    "支持图片格式": "JPG，JPEG，PNG，BMP",
    "应用商店": "支持",
    "base_main_sku_id": "100011431042",
    "base_item_url": "https://item.jd.com/100011431042.html",
    "base_item_third_cate_cd": "798"
}
```

## 命名实体识别

数据准备

| 训练集 | 44326 |
| ------ | ----- |
| 验证集 | 14775 |
| 测试集 | 14775 |

```shell
07/14/2021 14:25:17 - INFO - processors.ner_seq -   tokens: [CLS] 联 想 （ l e n o v o ） 扬 天 v 1 4 [UNK] 英 特 尔 酷 睿 [UNK] i 5 [UNK] 1 4 英 寸 窄 边 框 轻 薄 笔 记 本 电 脑 ( i 5 - 8 2 6 5 u [UNK] 8 g [UNK] 2 5 6 s s d [UNK] m x 1 1 0 [UNK] 2 g 独 显 ) [UNK] 太 空 灰 [UNK] 定 制 [SEP]
07/14/2021 14:25:17 - INFO - processors.ner_seq -   input_ids: 101 5468 2682 8020 154 147 156 157 164 157 8021 2813 1921 164 122 125 100 5739 4294 2209 6999 4729 100 151 126 100 122 125 5739 2189 4962 6804 3427 6768 5946 5011 6381 3315 4510 5554 113 151 126 118 129 123 127 126 163 100 129 149 100 123 126 127 161 161 146 100 155 166 122 122 121 100 123 149 4324 3227 114 100 1922 4958 4129 100 2137 1169 102 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
07/14/2021 14:25:17 - INFO - processors.ner_seq -   input_mask: 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
07/14/2021 14:25:17 - INFO - processors.ner_seq -   segment_ids: 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
07/14/2021 14:25:17 - INFO - processors.ner_seq -   label_ids: 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 3 7 7 3 6 1 4 4 4 4 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 3 6 7 7 7 7 7 7 7 7 7 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
```



### FocalLoss

Focal loss 是 文章 [Focal Loss for Dense Object Detection](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1708.02002) 中提出对简单样本的进行decay的一种损失函数。是对标准的Cross Entropy Loss 的一种改进。 F L对于简单样本（p比较大）回应较小的loss。 如论文中的图1， 在p=0.6时， 标准的CE然后又较大的loss， 但是对于FL就有相对较小的loss回应。这样就是对简单样本的一种decay。其中alpha 是对每个类别在训练数据中的频率有关， 但是下面的实现我们是基于alpha=1进行实验的。

<img src="/Volumes/yyx/学习/NLP知识体系/images/v2-464376ab6d4047cbb3a6d23872109377_720w-6081881.jpg" alt="img" style="zoom:50%;" />

标准的Cross Entropy 为：

<img src="/Volumes/yyx/学习/NLP知识体系/images/image-20210712172112323-6081673-6081675.png" alt="image-20210712172112323" style="zoom:67%;" />

Focal Loss 为：

<img src="/Volumes/yyx/学习/NLP知识体系/images/image-20210721215954290.png" alt="image-20210721215954290" style="zoom:50%;" />

<img src="/Volumes/yyx/学习/NLP知识体系/images/image-20210721220021487.png" alt="image-20210721220021487" style="zoom:50%;" />

其中 <img src="/Volumes/yyx/学习/NLP知识体系/images/image-20210721220051326.png" alt="image-20210721220051326" style="zoom:50%;" />

[Focal Loss_AI之路-CSDN博客_focal loss](https://blog.csdn.net/u014380165/article/details/77019084)

