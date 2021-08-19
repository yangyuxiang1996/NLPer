# jieba分词原理

[TOC]

## 中文分词方法归类

从20世纪80年代或更早的时候起，学者们研究了很多的分词方法，这些方法大致可以分为三大类：

1. 基于词表的方法
   * 正向最大匹配法(forward maximum matching method, FMM)
   * 逆向最大匹配法(backward maximum matching method, BMM)
2. 基于统计模型的分词方法
   * 基于N-gram语言模型的分词方法
   * 基于HMM的分词方法
   * 基于CRF的分词方法
   * 基于词感知机的分词方法
3. 基于深度学习的端到端的分词方法

jieba分词是中文分词工具中最常使用的工具之一，主要用到的分词方法是**基于N-gram语言模型的分词方法**和**基于HMM的分词方法**。本文首先对这两种方法进行总结，然后介绍jiebe分词的整体流程。

## 基于N-gram语言模型的分词方法

* 什么是语言模型？

简而言之，语言模型就是给定一个序列，计算这个序列出现的概率的模型。假如，给定一个单词序列$W=(w_0, w_1, ...,w_n)$，语言模型需要计算其出现的概率$p(W)$​，已知的概率分布来自于大量观测得到的语料。

以bi-gram为例：

<img src="/Volumes/yyx/学习/NLP知识体系/images/image-20210726232246685.png" alt="image-20210726232246685" style="zoom:50%;" />

假设随机变量S为一个汉字序列，W是S上所有可能的切分路径。对于分词，实际上就是求解使条件概率P(W|S)最大的切分路径$W^*$，即

<img src="/Volumes/yyx/学习/NLP知识体系/images/image-20210726232414189.png" alt="image-20210726232414189" style="zoom:50%;" />

根据贝叶斯公式：

<img src="/Volumes/yyx/学习/NLP知识体系/images/image-20210726232444921.png" alt="image-20210726232444921" style="zoom:50%;" />

这里，$P(S)$是一个归一化因子，$P(S|W)$恒等于1，因此，只需要求解$P(W)$，根据语言模型的公式计算即可，为了求解最大切分路径，这里可以使用动态规划（维特比算法）进行求解。

## 基于HMM的分词方法

使用n-gram语言模型进行中文分词需要足够多的语料库，并且不能够解决未登陆词的问题，这时就需要一个强大的统计机器学习模型，学习中文分词模式，并对未登陆词进行预测。HMM就是一种将分词转换成序列标注任务的统计学习模型，很好地解决了未登陆词问题。

<img src="/Volumes/yyx/学习/NLP知识体系/images/image-20210726234113034.png" alt="image-20210726234113034" style="zoom:50%;" />

假设每个词有四个词位：词首B，词中M，词尾E，单字成词S，这些词构成的序列成为观测序列X，其词位标注构成序列称为状态序列，也称为隐序列，根据HMM的两个假设：

1. 齐次马尔科夫性假设，即假设隐藏的马尔科夫链在任意时刻t的状态只依赖于其前一时刻的状态，与其它时刻的状态及观测无关，也与时刻t无关；
2. 观测独立性假设，即假设任意时刻的观测只依赖于该时刻的马尔科夫链的状态，与其它观测和状态无关

我们可以总结出HMM的计算公式：

<img src="/Volumes/yyx/学习/NLP知识体系/images/image-20210726234404691.png" alt="image-20210726234404691" style="zoom:50%;" />

式中：

* X：观测序列
* Y：隐藏状态序列
* $P(y_0)$：状态初始概率
* $P(y_t|y_{t-1})$：状态转移概率
* $P(x_t|y_t)$：状态发射概率

所以，HMM模型有三个基本问题：

- 概率计算问题，HMM的五元组，计算在模型下给定隐藏序列Y，计算观测序列X出现的概率也就是Forward-backward算法；
- 学习问题，已知观测序列{X}，隐藏序列{Y} ，估计模型的状态初始概率，状态转移概率和状态发射概率 ，使得在该模型下观测序列X的概率尽可能的大，即用极大似然估计的方法估计参数；
- 预测问题，也称为解码问题，已知模型状态初始概率，状态转移概率和状态发射概率和观测序列X，求最大概率的隐藏序列Y。

其中，jieba分词首先会使用大量的语料训练HMM生成状态初始概率矩阵，状态转移概率矩阵和状态发射概率矩阵，在分词的过程中直接使用开发者学习的到的矩阵参数，对待分词的序列进行预测，计算方法会涉及到维特比算法。

## jieba分词算法

结巴分词整体流程：

<img src="/Volumes/yyx/学习/NLP知识体系/images/image-20210727074047481.png" alt="image-20210727074047481" style="zoom:50%;" />



**算法要点**：

- 基于前缀词典实现高效的词图扫描，生成句子中汉字所有可能成词情况所构成的有向无环图 (DAG)
- 采用了动态规划查找最大概率路径, 找出基于词频的最大切分组合
- 对于未登录词，采用了基于汉字成词能力的 HMM 模型，使用了 Viterbi 算法

**分词模式**：

- 支持四种分词模式：
  - 精确模式，使用HMM，试图将句子最精确地切开，适合文本分析；
  - 全模式，把句子中所有的可以成词的词语都扫描出来, 速度非常快，但是不能解决歧义；
  - 搜索引擎模式，在精确模式的基础上，对长词再次切分，提高召回率，适合用于搜索引擎分词，cut_for_each。
  - paddle模式，利用PaddlePaddle深度学习框架，训练序列标注（双向GRU）网络模型实现分词。同时支持词性标注。

### 加载登陆词典

结巴分词中所用的词典保存在jieba/dict.txt中，如下所示，每一行包括三个字断，分别是单词，词频，词性：

```txt
AT&T 3 nz
B超 3 n
c# 3 nz
C# 3 nz
c++ 3 nz
C++ 3 nz
T恤 4 n
A座 3 n
```

### 建立前缀树词典

前缀词典构造的基本思路是将在统计词典中出现的每一个词的每一个前缀提取出来，统计词频，如果某个前缀词在统计词典中没有出现，词频统计为0，如果这个前缀词已经统计过，则不再重复。

```python
def gen_pfdict(file_name):
    lfreq = {}
    ltotal = 0
    f = open(file_name, 'r')
    for lineno, line in enumerate(f, 1):
        try:
            line = line.strip().decode('utf-8')
            word, freq = line.split(' ')[:2]
            freq = int(freq)
            lfreq[word] = freq
            ltotal += freq
            for ch in xrange(len(word)):
                wfrag = word[:ch + 1]
                if wfrag not in lfreq:
                    lfreq[wfrag] = 0
        except ValueError:
            raise ValueError(
                'invalid dictionary entry in %s at Line %s: %s' % (file_name, lineno, line))
    f.close()
    return lfreq, ltotal
```

举个例子，假如某一个统计词典中有如下几个词语，保存为my_dict.txt，我们这里先忽略词性：

```
我  123
在  234
学习  456
结巴  345
分词  456
结巴分词  23
学  2344
分  23
结 234
```

运行`gen_pfdict`得到：

```
我 123
在 234
学习 456
学 2344
结巴 345
结 234
分词 456
分 23
结巴分词 23
结巴分 0   // 结巴分这个前缀在词典中没有出现过，所以词频是0
```

### 句子分割

jieba分词中首先会对一个句子按照标点符号、非中文字符、空白符等进行分割，形成一系列子句。

```python
# 列举所有中文词中可能包含的字符
re_han_default = re.compile("([\u4E00-\u9FD5a-zA-Z0-9+#&\._%\-]+)", re.U)

# 列举空白符
re_skip_default = re.compile("(\r\n|\s)", re.U)

# 用于提取连续的汉字部分
re_han = re.compile("([\u4E00-\u9FD5]+)")

# 用于分割连续的非汉字部分
re_skip = re.compile("([a-zA-Z0-9\.]+(?:\.\d+)?%?)")
```

例如，我们想要分割如下句子：

```python
s = "“Microsoft”一词由“MICROcomputer（微型计算机）”和“SOFTware（软件）”两部分组成"

# 将连续出现的合法字符作为一个子句的划分
print(re_han_default.split(s))
>> ['“', 'Microsoft', '”', '一词由', '“', 'MICROcomputer', '（', '微型计算机', '）”', '和', '“', 'SOFTware', '（', '软件', '）”', '两部分组成', '']
```

### 建立分词DAG图

```python
def get_DAG(self, sentence):
    self.check_initialized()
    DAG = {}
    N = len(sentence)
    for k in xrange(N):
        tmplist = []
        i = k
        frag = sentence[k]
        while i < N and frag in self.FREQ:
            if self.FREQ[frag]:
                tmplist.append(i)
            i += 1
            frag = sentence[k:i + 1]
        if not tmplist:
            tmplist.append(k)
        DAG[k] = tmplist
    return DAG
```

假设我们现在有一句子："我在学习结巴分词"，对这一句子构造DAG图可得如下数据，其中0-7表示这句话中的第i个字的索引，其对应的value表示以该字开头的可能的词的结束的位置

```
0 [0]
1 [1]
2 [2, 3]
3 [3]
4 [4, 5, 7]
5 [5]
6 [6, 7]
7 [7]
```

为了加深对DAG图的理解，我们将其可视化，如下图所示：

<img src="/Volumes/yyx/学习/NLP知识体系/images/image-20210727232650024.png" alt="image-20210727232650024" style="zoom:50%;" />

从DAG图我们可以看出，从这句话的开头“我”到结束“词”，一共有以下10种：

```text
我/在/学/习/结/巴/分/词
我/在/学习/结巴分词
我/在/学习/结/巴/分/词
我/在/学习/结巴/分词
我/在/学习/结/巴/分词
我/在/学习/结巴/分/词
我/在/学/习/结/巴/分词
我/在/学/习/结巴/分/词
我/在/学/习/结巴分词
我/在/学/习/结巴/分词
```

### 动态规划寻找最佳切分组合

jieba分词中使用uni-gram语言模型计算每一条切分路径的概率值：

<img src="/Volumes/yyx/学习/NLP知识体系/images/image-20210727233537343.png" alt="image-20210727233537343" style="zoom:50%;" />

我们只需要计算每一条切分路径的概率值，选择概率最大的那一条切分路径即可。

<img src="/Volumes/yyx/学习/NLP知识体系/images/image-20210727233618703.png" alt="image-20210727233618703" style="zoom:50%;" />

为了使得计算方便，每一个词出现的概率等于该词在前缀里的词频除以所有词的词频之和。如果词频为0或是不存在，当做词频为1来处理。这里会取对数概率，即在每个词概率的基础上取对数，一是为了防止下溢，二后面的概率相乘可以变成相加计算。
$$
P(w_n)=\log \frac{freq[w_n]+1}{total}
$$
最后使用动态规划算法算出概率最大的路径。

```python
import math
def calc(freq, total, sentence, DAG, route):
    N = len(sentence)
    route[N] = (0, 0)
    logtotal = math.log(total)
    for idx in range(N - 1, -1, -1):
        route[idx] = max((math.log(freq.get(sentence[idx:x + 1], 0) + 1) -
                            logtotal + route[x + 1][0], x) for x in DAG[idx])
        
def __cut_DAG_NO_HMM(sentence):
    lfreq, ltotal = gen_pfdict(file_name='my_dict.txt')
    DAG = get_DAG(lfreq, sentence)
    route = {}
    calc(lfreq, ltotal, sentence, DAG, route)
    x = 0
    N = len(sentence)
    buf = ''
    re_eng = re.compile('[a-zA-Z0-9]', re.U)
    while x < N:
        y = route[x][1] + 1
        l_word = sentence[x:y]
        if re_eng.match(l_word) and len(l_word) == 1:
            buf += l_word
            x = y
        else:
            if buf:
                yield buf
                buf = ''
            yield l_word
            x = y
    if buf:
        yield buf
        buf = ''

sentence = "我在学习结巴分词"
sentence = __cut_DAG_NO_HMM(sentence)
print(" ".join(sentence))
>> 我 在 学习 结巴 分词
```

### HMM解决未登陆词

在jieba分词中，基于HMM的分词主要是作为基于uni-gram分词的一个补充，主要是解决OOV（out of vocabulary）问题的，它的作用是对未登录词典的词进行识别发现。我们首先用一个例子说明HMM的重要性。比如我们要对一个包含人名的句子进行分词，“韩冰是个好人”。“韩冰”这个词不在词典之中，所以前面基于词典+uni-Gram语言模型的方法进行分词就会将“韩冰”这个人名分成“韩”+“冰”。所以我们需要一个有一定泛化能力的机器学习模型对这些新词进行发现。

```python
sentence = "韩冰是个好人"
sl = __cut_DAG_NO_HMM(sentence)
print(" ".join(sl))

>> 韩 冰 是 个 好人

# 采用jieba分词的精确模式，使用HMM模型
print(" ".join(jieba.cut(sentence, HMM=True)))
>> 韩冰 是 个 好人
```

使用HMM进行分词的原理在前面已经介绍过了。**利用HMM模型进行分词，主要是将分词问题视为一个序列标注（sequence labeling）问题，其中，句子为观测序列，分词结果为状态序列。首先通过语料训练出HMM相关的模型，然后利用Viterbi算法进行求解，最终得到最优的状态序列，然后再根据状态序列，输出分词结果。**

在中文分词中，状态序列的标注有四个：

* "B":Begin（这个字处于词的开始位置） 
* "M":Middle（这个字处于词的中间位置）
* "E":End（这个字处于词的结束位置）
* "S":Single（这个字是单字成词）

要统计的主要有三个概率表：1)位置转换概率，即B，M，E，S四种状态的转移概率；2）位置到单字的发射概率，比如P("和"|M)表示一个词的中间出现”和"这个字的概率；3) 词语以某种状态开头的概率，其实只有两种，要么是B，要么是S。

jieba分词中已经包含开发者预训练好的HMM模型，所以在中文分词的过程中，我们只需要加载训练好的数据即可，例如prob_trans.py这个文件，保存了状态转移概率：

```json
{'B': {'E': -0.510825623765990, 'M': -0.916290731874155},
 'E': {'B': -0.5897149736854513, 'S': -0.8085250474669937},
 'M': {'E': -0.33344856811948514, 'M': -1.2603623820268226},
 'S': {'B': -0.7211965654669841, 'S': -0.6658631448798212}}
```

P(E|B) = -0.510, P(M|B) =-0.916，说明当我们处于一个词的开头时，下一个字是结尾的概率要高于下一个字是中间字的概率，符合我们的直觉，因为二个字的词比多个字的词更常见。P(M|E)=0，说明这个转移是不可能存在的。

### Viterbi算法

viterbi维特比算法解决的是篱笆型的图的最短路径问题，图的节点按列组织，每列的节点数量可以不一样，每一列的节点只能和相邻列的节点相连，不能跨列相连，节点之间有着不同的距离。

<img src="/Volumes/yyx/学习/NLP知识体系/images/image-20210728102310137.png" alt="image-20210728102310137" style="zoom:50%;" />

为了找出Start到End之间的最短路径，我们先从Start开始从左到右一列一列地来看。首先起点是Start，从Start到“韩”字对应的状态列的路径有四种可能：Start-B、Start-E、Start-M，Start-S。对应的路径长度即：

<img src="/Volumes/yyx/学习/NLP知识体系/images/image-20210728102334292.png" alt="image-20210728102334292" style="zoom:50%;" />

我们不能武断地说这四条路径中中的哪一段必定是全局最短路径中的一部分，目前为止任何一段都有可能是全局最优路径的备选项。我们继续往右看，到了“冰”这一列列。按照四个状态进行逐一分析，先看到达“冰”(B)节点的各个路径长度。

<img src="/Volumes/yyx/学习/NLP知识体系/images/image-20210728102403514.png" alt="image-20210728102403514" style="zoom:50%;" />

以上这四条路径，各节点距离加起来对比一下，我们就可以知道其中哪一条是最短的。因为Start-B-B是最短的，那么我们就知道了经过“冰”(B)的所有路径当中Start-B-B是最短的，其它三条路径路径都比Start-B-B长，绝对不是目标答案，可以大胆地删掉了。删掉了不可能是答案的路径，就是viterbi算法（维特比算法）的重点，因为后面我们再也不用考虑这些被删掉的路径了。现在经过“冰”(B)的所有路径只剩一条路径了(红色标识)，以此类推，我们可以分别找出到达“冰”字对应列的所有四个状态的最优路径。

<img src="/Volumes/yyx/学习/NLP知识体系/images/image-20210728102442004.png" alt="image-20210728102442004" style="zoom:50%;" />

整体代码：

```python
def load_model():
    start_p = pickle.load(get_module_res("finalseg", PROB_START_P)) # 加载初始状态概率矩阵，start->BMES
    trans_p = pickle.load(get_module_res("finalseg", PROB_TRANS_P)) # 加载转移概率矩阵
    emit_p = pickle.load(get_module_res("finalseg", PROB_EMIT_P))  # 加载发射概率矩阵
    return start_p, trans_p, emit_p

def viterbi(obs, states, start_p, trans_p, emit_p):
    # 维特比算法进行预测
    # obs：观测序列，如“我爱中国”
    V = [{}]  # tabular
    path = {}
    for y in states:  # init，'BMES'
        V[0][y] = start_p[y] + emit_p[y].get(obs[0], MIN_FLOAT)  # 对数概率，可以直接相加
        path[y] = [y]
    for t in range(1, len(obs)):
        V.append({})
        newpath = {}
        for y in states:
            em_p = emit_p[y].get(obs[t], MIN_FLOAT)
            # 取概率最大的那一条路径
            (prob, state) = max(
                [(V[t - 1][y0] + trans_p[y0].get(y, MIN_FLOAT) + em_p, y0) for y0 in PrevStatus[y]]) 
            V[t][y] = prob
            newpath[y] = path[state] + [y]
        path = newpath

    (prob, state) = max((V[len(obs) - 1][y], y) for y in 'ES')  # 转移到end

    return (prob, path[state])

def __cut(sentence):
    global emit_P
    prob, pos_list = viterbi(sentence, 'BMES', start_P, trans_P, emit_P)
    begin, nexti = 0, 0
    # print pos_list, sentence
    for i, char in enumerate(sentence):
        pos = pos_list[i]
        if pos == 'B':
            begin = i
        elif pos == 'E':
            yield sentence[begin:i + 1]
            nexti = i + 1
        elif pos == 'S':
            yield char
            nexti = i + 1
    if nexti < len(sentence):
        yield sentence[nexti:]
      
def __cut_DAG(self, sentence):
    DAG = self.get_DAG(sentence)
    route = {}
    self.calc(sentence, DAG, route)
    x = 0
    buf = ''
    N = len(sentence)
    while x < N:
        y = route[x][1] + 1
        l_word = sentence[x:y]
        if y - x == 1:
            buf += l_word
        else:
            if buf:
                if len(buf) == 1:
                    yield buf
                    buf = ''
                else:
                    if not self.FREQ.get(buf):  # 如果前缀词典中没有这个词语，利用HMM进行精确分词
                        recognized = finalseg.cut(buf)
                        for t in recognized:
                            yield t
                    else:
                        for elem in buf:  # 如果前缀词典中有这个词语，直接输出
                            yield elem
                    buf = ''
            yield l_word
        x = y

    if buf:
        if len(buf) == 1:
            yield buf
        elif not self.FREQ.get(buf):
            recognized = finalseg.cut(buf)
            for t in recognized:
                yield t
        else:
            for elem in buf:
                yield elem
```

[Jieba分词原理解析 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/245372320)







