# 关键词抽取

关键词抽取，也可以称为文本标签抽取，例如，“今天我想吃烧烤”，“烧烤”一词就可以看作是这段文本的关键词，或者说是这一段文本的标签，这个标签往往蕴含了文本最重要的信息，能够在一些下游任务中发挥一定的作用。比如，“烧烤”应用在文本分类任务中，可以将这段文本和“美食”联系起来，如果应用在推荐系统中，可以根据“烧烤”这个标签做一路召回，然后在做排序。

关键词提取的常见方法：

<img src="/Volumes/yyx/学习/NLP知识体系/images/image-20210817170048973.png" alt="image-20210817170048973" style="zoom:50%;" />

关键词是代表文章重要内容的一组词。对文本聚类、分类、自动摘要等起重要作用。此外，它还能使人们便捷地浏览和获取信息。 类似于其他的机器学习方法，关键词提取算法一般也可以分为有监督和无监督两类:

- 有监督的关键词提取方法主要是通过分类的方式进行，通过构建一个较为丰富和完善的词表，然后通过判断每个文档与词表中每个词的匹配程度， 以类似打标签的方式，达到关键词提取的效果。
  - 有监督的方法能够获取到较高的精度，但缺点是需要大批量的标注数据，人工成本过高
  - 另外，现在每天的信息增加过多，会有大量的新信息出现，一个固定的词表有时很难将新信息的内容表达出来， 但是要人工维护这个受控的词表却要很高的人力成本，这也是使用有监督方法来进行关键词提取的一个比较大的缺陷

* 无监督的方法对数据的要求比较低，既不需要一张人工生成、维护的词表，也不需要人工标准语料辅助进行训练。 因此，这类算法在关键词提取领域的应用更受到大家的青睐。

  * TF-IDF 算法

  * TextRank 算法
  * 主题模型算法
    * LSA
    * LSI
    * LDA



## 基于TF-IDF的关键词抽取

### tfidf算法

 TF-IDF 算法(Term Frequency-Inverse Document Frequency，词频-逆文档频次算法)，是一种基于统计的计算方法， 常用于评估一个文档集中一个词对某份文档的重要程度。这种作用显然很符合关键字抽取的需求，一个词对文档越重要，那就越可能 是文档对的关键词，常将 TF-IDF 算法应用于关键词提取中，作为baseline使用。

公式：
$$
tf \times idf(i, j) = {tf}_{ij} \times {idf}_{i} = \frac{n_{ij}}{\sum_{k} n_{kj}} \times log\Big(\frac{|D|}{1+|D_{i}|}\Big)
$$
含义：

<img src="/Volumes/yyx/学习/NLP知识体系/images/image-20210817173721125.png" alt="image-20210817173721125" style="zoom:50%;" />

- TF 算法是统计一个词在一篇文档中出现的频次，其基本思想是，一个词在文档中出现的次数越多，则其对文档的表达能力就越强
- IDF 算法则是统计一个词在文档集的多少个文档中出现，其基本思想是，如果一个词在越少的文档中出现，则其对文档的区分能力也就越强

### 代码

```python
class keywordExtractor():
    def __init__(self):
        idf_dict = {}
        data_list = []
        with open("./idf.txt") as f:
            for line in f:
                ll = line.strip().split(" ")
                if len(ll) != 2:
                    continue
                if ll[0] not in idf_dict:
                    idf_dict[ll[0]] = float(ll[1])
                data_list.append(float(ll[1]))
        self.__idf_dict = idf_dict
        self.median = np.median(data_list)
        self.stop_words = self.load_stop_words("./stop_words.txt")

    def load_stop_words(self, path):
        print("加载停用词")
        stop_words = []
        with open(path, "r") as f:
            for line in f.readlines():
                line = line.strip()
                stop_words.append(line)
        return stop_words
        
    def get_idf(self,word):
        return self.__idf_dict.get(word, self.median)

    def clean_data(self, sent, sep='<'):
        '''
        @description: 过滤无用符号，假如前后空格，避免影响分词结果
        @param {type}
        sent: 句子
        sep: 分隔符是以< or [ 开头
        @return: string 清洗后的句子
        '''
        sent = re.sub(
            r"[\s+\.\!\/_,$%^*(+\"\')]+|[+——()?【】“”！，。？、~@#￥%……&*（）：]+", "", sent)
        return sent
    
    def predict(self, query, top_n = 1, withWeight=True):
        if len(query) <= 2:
            return [query]
    
        # 切词
        query = self.clean_data(query)
        word_list = list(jieba.cut(query))

        # 去除停用词
        word_list = [word for word in word_list if word not in self.stop_words and len(word) >= 2]
        # print(word_list)
        tf = Counter(word_list)
        n = sum(tf.values())
        
        if len(word_list) < top_n:
            return word_list
        
        # 默认赋值
        idf_list = []
        word_list = list(set(word_list))
        for word in word_list:
            idf_list.append(self.get_idf(word)*tf[word]/n)
        
        zip_list = zip(range(len(idf_list)), idf_list)
        n_large_idx = [i[0] for i in heapq.nlargest(top_n, zip_list, key=lambda x:x[1])]

        if withWeight:
            return [(word_list[i], idf_list[i]) for i in n_large_idx]
        else:
            return [word_list[i] for i in n_large_idx]
```

 测试：

```python
keyword_extractor = keywordExtractor()
query = "原标题：香港新增2例输入新冠肺炎确诊病例香港特区政府卫生署卫生防护中心8月14日公布，截至当日零时，香港新增2例新冠肺炎确诊病例，均为输入病例。目前，香港累计报告新冠肺炎确诊病例12032例。（总台记者）"
print(keyword_extractor.predict(query, 5))
```

结果：

`[('病例', 0.8587752282844444), ('肺炎', 0.7359811208133333), ('确诊', 0.6721555762425), ('新冠', 0.6641537501611111), ('香港', 0.484342987225)]`

直接拿jieba分词库中的关键词提取接口：

```python
from jieba import analyse
print(analyse.extract_tags(query, 5, withWeight=True))
```

结果：

`[('病例', 0.8587752282844444), ('肺炎', 0.7359811208133333), ('确诊', 0.6721555762425), ('新冠', 0.6641537501611111), ('香港', 0.484342987225)]`

**结果一致！**



## 基于TextRank的关键词提取

### PageRank

PageRank (PR) 是一种用于计算网页权重的算法。我们可以把所有的网页看成一个大的有向图。在这个图中，一个节点就是一个网页。如果网页 A 有到网页 B 的链接，它可以表示为从 A 到 B 的有向边。

在我们构建了整个图之后，我们可以通过以下公式为网页分配权重：
$$
S(V_i)=(1-d)+d*\sum_{j\in In(v_i)}{\frac{1}{|Out(V_j)|}S(v_j)}
$$
式中：

* $S(V_i)$：第i个webpage的权重
* $d$：阻尼系数，防止没有外部连接
* $In(V_i)$：page i的入站链接集合（链接到i的网页组成的集合）
* $Out(V_j)$： page j的出站链接集合（从网页j出去的网页组成的集合）
* $|Out(V_j)|$：page j的出站链接数量

举个例子：

<img src="/Volumes/yyx/学习/NLP知识体系/images/image-20210818162457035.png" alt="image-20210818162457035" style="zoom: 33%;" />

图中每一个节点代表一个webpage，我们现在想要计算webpage e的权重，首先计算公式中的求和部分，展开如下：

<img src="/Volumes/yyx/学习/NLP知识体系/images/image-20210818162924682.png" alt="image-20210818162924682" style="zoom:50%;" />

因此，webpage e的权重更新公式如下：

<img src="/Volumes/yyx/学习/NLP知识体系/images/image-20210818163030554.png" alt="image-20210818163030554" style="zoom: 33%;" />

* **不难看出，每个webpage的权重由其入站链接决定，如果入站链接的权重较高的话，相应的就会提高该page的权重，例如，如果page a引用了page e，如果page a具有较高的权重，那么page e的权重也会提高；同时，一个网站，如果越多的网站链接到它，这个网站的权重也会越高；**
* **如果某一个网页没有被引用的话，它的权重就是1-d，控制权重不为0；**
* **为什么要除以$|Out(V_j)|$，可以理解为$\frac{1}{|Out(V_j)|}$是一个权重，当某一个网页链接到别的网页的数量越大时，在计算别的网页的权重时，这个网页的权重就应该降低一些，就好比投票，如果一个人投了很多票，那么这个人投票的重要度也就降低了。除以$|Out(V_j)|$就相当于做了一个归一化操作。**
* **加入阻尼因子的作用：防止Dead Ends问题，如上图中的a，a不存在链接到a的网页，如果不用阻尼因子进行平滑的话，权重最终会收敛到0，这是需要避免的情况**
* **求S的过程，实际是一个马尔科夫收敛过程。**

[Understand TextRank for Keyword Extraction by Python | by Xu LIANG | Towards Data Science](https://towardsdatascience.com/textrank-for-keyword-extraction-by-python-c0bae21bcec0)

### TextRank

TextRank是从PageTank的思想上发展起来的，TextRank和PageTank有什么区别？简单的回答就是PageRank是网页排名，TextRank是文字排名。TextRank中的单词就可以看作是 PageRank中的网页，词就是Graph中的节点，而词与词之间的边，则利用“共现”关系来确定。所谓“共现”，就是共同出现，即在一个给定大小的滑动窗口内的词，认为是共同出现的，而这些单词间也就存在着边，所以基本思路是一样的。

假设我们有一个n个word的句子，
$$
w_1,w_2,w_3,...,w_n
$$
设定window size为k，有窗口：$$[w_1, w_2, …, w_k], [w_2, w_3, …, w_{k+1}], [w_3, w_4, …, w_{k+2}]$$。

假设有一句话，经过处理后得到候选单词序列：`[时间，流浪，地球，感觉，复古，时代，电影制作]`，

设置窗口大小k=4，所以我们得到4个窗口：

`[时间，流浪，地球，感觉]`，`[流浪，地球，感觉，回归]`，`[地球，感觉，回归，时代]`，`[感觉，回归，时代，电影制作]`。

对于窗口`[时间，流浪，地球，感觉]`，任何两个词对都有一个无向边。

所以我们得到`（时间，流浪），（时间，地球），（时间，感觉），（流浪，地球），（流浪，感觉），（地球，感觉）`。

将这个pair对组成的集合构建成无向图，就可以应用TextRank算法计算每一个word的权重，从而提取关键词。

<img src="/Volumes/yyx/学习/NLP知识体系/images/image-20210818172144463.png" alt="image-20210818172144463" style="zoom:50%;" />

不难发现，相对于PageRank里的无权有向图，这里建立的是无权无向图，原论文中对于关键词提取任务主要也是构建的无向无权图，对于有向图，论文提到是基于词的前后顺序角度去考虑，即给定窗口，比如对于“长裙”来说，“淡黄”与它之间是入边，而“蓬松”与它之间是出边，但是效果都要比无向图差。

<img src="/Volumes/yyx/学习/NLP知识体系/images/image-20210818175611398.png" alt="image-20210818175611398" style="zoom:50%;" />

### 代码

逻辑：

<img src="/Volumes/yyx/学习/NLP知识体系/images/image-20210819185834187.png" alt="image-20210819185834187" style="zoom:50%;" />



```python
import jieba.posseg
import jieba.analyse
import numpy as np
import heapq


class TextRank(object):
    def __init__(self):
        self.d = 0.85
        self.stop_words = self.load_stop_words("./stop_words.txt")
        self.iter = 10
        self.word2id = {}
        self.id2word = {}
        self.tokenizer = jieba.posseg.dt
        self.allowPOS = ['ns', 'n', 'vn', 'v']
        self.window_size = 5
        self.graph = []
        self.eps = 1e-6

    def load_stop_words(self, path):
        print("加载停用词")
        stop_words = set()
        with open(path, "r") as f:
            for line in f.readlines():
                line = line.strip()
                stop_words.add(line)
        return stop_words

    def tokenize(self, sentence):
        return self.tokenizer.cut(sentence)

    def pair_filter(self, wp):
        return wp.flag in self.allowPOS \
                and len(wp.word.strip()) >= 2 \
                and wp.word not in self.stop_words

    def extract_tags(self, sentence, topK=20, withWeight=False, allowPOS=('ns', 'n', 'vn', 'v'), withFlag=False):
        if allowPOS is not None:
            self.allowPOS = allowPOS
        
        sentence = list(self.tokenize(sentence))
        clean_sent = [wp for wp in sentence if self.pair_filter(wp)]
        for wp in clean_sent:
            if wp.word not in self.word2id:
                self.word2id[wp.word] = len(self.word2id)
    
        self.id2word = {i: word for word, i in self.word2id.items()}
        self.graph = np.zeros(shape=(len(self.word2id), len(self.word2id)))
        self.ws = np.ones(len(self.word2id)) / len(self.word2id)

        for i, wp in enumerate(sentence):
            if wp.word in self.word2id:
                a = self.word2id[wp.word]
                for j in range(i+1, i+self.window_size):
                    if j >= len(sentence):
                        break
                    if sentence[j].word in self.word2id:
                        b = self.word2id[sentence[j].word]
                        self.graph[a][b] += 1
                        self.graph[b][a] += 1
        
        self.graph = self.graph / (np.sum(self.graph, axis=0) + self.eps)
        for i in range(self.iter):
            self.ws = (1 - self.d) + self.d * np.dot(self.graph, self.ws)  
            
        max_ws, min_ws = np.max(self.ws), np.min(self.ws)
        self.ws = (self.ws - min_ws / 10.0) / (max_ws - min_ws / 10.0)

        # 取topK
        zip_list = zip(range(len(self.ws)), self.ws)
        n_large_idx = [i[0] for i in heapq.nlargest(topK, zip_list, key=lambda x:x[1])]

        if withWeight:
            return [(self.id2word[i], self.ws[i]) for i in n_large_idx]
        else:
            return [self.id2word[i] for i in n_large_idx]


if __name__ == '__main__':
    query = "原标题：香港新增2例输入新冠肺炎确诊病例香港特区政府卫生署卫生防护中心8月14日公布，截至当日零时，香港新增2例新冠肺炎确诊病例，均为输入病例。目前，香港累计报告新冠肺炎确诊病例12032例。（总台记者）"
    tr = jieba.analyse.TextRank()
    print(tr.extract_tags(sentence=query, topK=5, withWeight=True))
    tr = TextRank()
    print(tr.extract_tags(sentence=query, topK=5, withWeight=True))
```

结果：

直接使用结巴库：

`[('病例', 1.0), ('肺炎', 0.8549361906523198), ('香港', 0.8125918070357397), ('确诊', 0.8012586985820523), ('新冠', 0.676185135476088)]`

代码运行结果：

`[('病例', 1.0), ('肺炎', 0.8413701152270399), ('确诊', 0.8027702275390493), ('香港', 0.6776252276262762), ('新冠', 0.6765789656391807)]`

结果相差不多！

和tfidf相比：

`[('病例', 0.8587752282844444), ('肺炎', 0.7359811208133333), ('确诊', 0.6721555762425), ('新冠', 0.6641537501611111), ('香港', 0.484342987225)]`

关键词提取结果差不多。

### 小结

**优点：**

1） 无监督方式，无需构造数据集训练。

2） 算法原理简单且部署简单。

3） 继承了PageRank的思想，效果相对较好，相对于TF-IDF方法，可以更充分的利用文本元素之间的关系。

**缺点：**

1） 结果受分词、文本清洗影响较大，即对于某些停用词的保留与否，直接影响最终结果。

2） 虽然与TF-IDF比，不止利用了词频，但是仍然受高频词的影响，因此，需要结合词性和词频进行筛选，以达到更好效果，但词性标注显然又是一个问题。



## 基于LDA的关键词提取