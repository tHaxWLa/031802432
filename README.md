
###[GITHUB传送门](https://github.com/tHaxWLa/031802432)

###余弦定义
两个向量间的余弦值可以通过欧几里得点积公式求出:
![](https://img2020.cnblogs.com/blog/2145430/202009/2145430-20200917134857115-2052157612.png)
给定两个属性向量A和B,真余弦相似性θ由点积和向量长度给出，如下所示:
![](https://img2020.cnblogs.com/blog/2145430/202009/2145430-20200917135025203-2137984782.png)
这里的Ai,Bi分别代表向量A和B的各[分向量](https://baike.baidu.com/item/%E5%88%86%E5%90%91%E9%87%8F/3729375?fr=aladdin).

###如果对如何用余弦实现文本相似度感兴趣的,可以去看这篇博[文本相似度的衡量之余弦相似度](https://www.cnblogs.com/zhangyafei/p/10617237.html)

话不多说~~,开始走流程~~

###文本相似度计算的处理流程

![](https://img2020.cnblogs.com/blog/2145430/202009/2145430-20200917160841215-242373419.png)

###PSP表格

 <table>
        <tr>
            <th>PSP2.1</th>
            <th>Personal Software Process Stages</th>
            <th>预估耗时（分钟）</th>
            <th>实际耗时（分钟）</th>
        </tr>
        <tr>
            <th>Planning</th>
            <th>计划</th>
            <th>16</th>
            <th>16</th>
        </tr>
        <tr>
            <th> Estimate</th>
            <th>估计这个任务需要多少时间</th>
            <th>16</th>
            <th>8</th>
        </tr>
        <tr>
            <th>Development</th>
            <th>开发</th>
            <th>512</th>
            <th>770</th>
        </tr>
        <tr>
            <th>Analysis</th>
            <th>需求分析 (包括学习新技术)</th>
            <th>256</th>
            <th>170</th>
        </tr>
        <tr>
            <th>Design Spec</th>
            <th>生成设计文档</th>
            <th>32</th>
            <th>16</th>
        </tr>
        <tr>
            <th>Design Review</th>
            <th>设计复审</th>
            <th>32</th>
            <th>8</th>
        </tr>
        <tr>
            <th>Coding Standard</th>
            <th>代码规范 (为目前的开发制定合适的规范)</th>
            <th>8</th>
            <th>8</th>
        </tr> 
        <tr>
            <th>Test</th>
            <th>测试（自我测试，修改代码，提交修改）</th>
            <th>128</th>
            <th>180</th>
        </tr>
        <tr>
            <th>Reporting</th>
            <th>报告</th>
            <th>128</th>
            <th>90</th>
        </tr>
        <tr>
            <th>Test Repor</th>
            <th>测试报告</th>
            <th>64</th>
            <th>50</th>
        </tr>
        <tr>
            <th>Size Measurement</th>
            <th>计算工作量</th>
            <th>32</th>
            <th>40</th>
        </tr>
        <tr>
            <th> Postmortem & Process Improvement Plan</th>
            <th> 事后总结, 并提出过程改进计划</th>
            <th>64</th>
            <th>15</th>
        </tr>
        </tr>
        	<th></th>
        	<th>合计</th>
        	<th>1288</th>
        	<th>1371</th>
        </tr>
    </table>

##先分成四个模块
###词性处理的部分
```python
    @staticmethod
    def extract_keyword(content):  # 提取关键词
        # 正则过滤 html 标签
        re_exp = re.compile(r'(<style>.*?</style>)|(<[^>]+>)', re.S)
        content = re_exp.sub(' ', content)
        # html 转义符实体化
        content = html.unescape(content)
        # 切割
        seg = [i for i in jieba.cut(content, cut_all=True) if i != '']
        # 提取关键词
        keywords = jieba.analyse.extract_tags("|".join(seg), topK=200, withWeight=False)
        return keywords
```

###[oneHot](https://baike.baidu.com/item/%E7%8B%AC%E7%83%AD%E7%A0%81/1428731?fr=aladdin)
one-hot是比较常用的文本特征特征提取的方法。
one-hot编码，又称“独热编码”。其实就是用N位状态寄存器编码N个状态，每个状态都有独立的寄存器位，且这些寄存器位中只有一位有效，说白了就是只能有一个状态。
```python
    @staticmethod
    def one_hot(word_dict, keywords):  # oneHot编码
        # cut_code = [word_dict[word] for word in keywords]
        cut_code = [0]*len(word_dict)
        for word in keywords:
            cut_code[word_dict[word]] += 1
        return cut_code
```

###逻辑处理部分
```python
    def main(self):
        # 提取关键词
        keywords1 = self.extract_keyword(self.s1)
        keywords2 = self.extract_keyword(self.s2)
        # 词的并集
        union = set(keywords1).union(set(keywords2))
        # 编码
        word_dict = {}
        i = 0
        for word in union:
            word_dict[word] = i
            i += 1
        # oneHot编码
        s1_cut_code = self.one_hot(word_dict, keywords1)
        s2_cut_code = self.one_hot(word_dict, keywords2)
        # 余弦相似度计算
        sample = [s1_cut_code, s2_cut_code]
        # 除零处理
        try:
            sim = cosine_similarity(sample)
            return sim[1][0]
        except Exception as e:
            print(e)
            return 0.0
```
###文件的读取和输出
```python
if __name__ == '__main__':
    path1=sys.argv[1]	#原文文件
    path2=sys.argv[2]	#抄袭版论文的文件
    path3=sys.argv[3]	#答案文件

f = open(path1,encoding='utf-8')   #读取原文文件
s1 = f.read()
f.close()
f = open(path2,encoding='utf-8')	#读取抄袭版论文的文件
s2 = f.read()
f.close()

similarity = CosineSimilarity(s1,s2)
result = round(similarity.main(),2)

with open(path3,"a",encoding='utf-8') as f:	
		f.write(str(result))	#输出到答案文件
```

##时间分析
![](https://img2020.cnblogs.com/blog/2145430/202009/2145430-20200917152341392-379756867.png)

##样例的相似度

<table>
        <tr>
            <th>文本名称</th>
            <th>相似度</th>
        </tr>
        <tr>
            <th>orig.txt</th>
            <th>1.00</th>
        </tr>
        <tr>
            <th>orig_0.8_add.txt</th>
            <th>0.84</th>
        </tr>
        <tr>
            <th>orig_0.8_del.txt</th>
            <th>0.73</th>
        </tr>
        <tr>
            <th>orig_0.8_dis_1.txt</th>
            <th>0.89</th>
        </tr>
        <tr>
            <th>orig_0.8_dis_3.txt</th>
            <th>0.87</th>
        </tr>
        <tr>
            <th>orig_0.8_dis_7.txt</th>
            <th>0.84</th>
        </tr>
        <tr>
            <th>orig_0.8_dis_10.txt</th>
            <th>0.75</th>
        </tr>
        <tr>
            <th>orig_0.8_dis_15.txt</th>
            <th>0.63</th>
        </tr>
        <tr>
            <th>orig_0.8_mix.txt</th>
            <th>0.80</th>
        </tr>
        <tr>
            <th>orig_0.8_rep.txt</th>
            <th>0.74</th>
        </tr>
</table>

