同义词挖掘
========



**同义词挖掘方法**：</br>

（1）百度百科同义词 </br>
（2）word2vector </br>
（3）语义共现网络的节点相似度 </br>
（4）levenshtein距离 </br>
（5）DPE模型（undo）[【参考文献】](https://arxiv.org/pdf/1706.08186.pdf) </br>

主要功能
========
###  1. 百度百科同义词
--------
<img src="https://github.com/tigerchen52/synonym_detection/blob/master/input/img/baike2.png"  width="400" /> <img src="https://github.com/tigerchen52/synonym_detection/blob/master/input/img/baike.png"  width="400" /> </br>

* 如上图所示，是在百度百科中搜索“凤梨”返回的页面结果，左边这个图为凤梨的description，右边这个图为凤梨的info box。
* description中有这么一句话“原产美洲热带地区。俗称菠萝，为著名热带水果之一。”，那么我们可以把凤梨“俗称”菠萝提取出来就到了同义词 </br>
* info box中有“别称”、“英文名称”、“又称”等属性，我们同样可以当做同义词提取出来，这样就完成了同义词的挖掘</br>

代码示例
```python
def baike_invoke():
    import baike_crawler_model
    print(baike_crawler_model.baike_search(('凤梨', '001')))

if __name__ == '__main__':
    baike_invoke()
```
输出:

    ['菠萝皮', '地菠萝', '菠萝', '草菠萝']

###  2. word2vector
--------
* 使用gensim包对输入的语料训练词向量，然后计算词向量之间余弦相似度，返回top-k个词作为同义词 </br>

代码示例
```python
python synonym_detect -corpus_path  ../input/三体.txt -input_word_path ../temp/input_word.txt -process_number 2 if_use_w2v_model True
```
参数
* -corpus_path 为语料文件，使用三体小说作为训练语料
* -input_word_path 输入词表，对词表中的词进行同义词挖掘
* -process_number 2 进程数量
* -if_use_w2v_model True 使用word2vector模型

输人 input_word:
```
1|海王星
2|女孩
3|椅子
4|海军
5|阵列
6|变化
7|程心
8|火焰
9|天空
10|建造
```
输出 :
```
1	海王星	海王星|土星|天王星|背面|金星
3	椅子	椅子|办公桌|地板|地毯|铁锹
2	女孩	女孩|中年人|女孩儿|女子|泪光
9	天空	天空|晨光|夜空|暮色|漆黑
4	海军	海军|军种|服役|事务性|政工
6	变化	变化|隐隐约约|异常|微妙|所致
5	阵列	阵列|矩形|一千公里|环|标示出
7	程心	程心|AA|艾|当程心|曹彬
8	火焰	火焰|暗红|山脉|灼热|变幻
10	建造	建造|天梯|最小|准|航空母舰
```

###  3. 语义共现网络的节点相似度
语义共现网络本质是根据上下文构建的图，图中的节点是词，边是这个词的上下文相关词。对于语义共现网络的两个节点，如果这两个节点的共同邻居节点越多，说明这两个词的上下文越相似，是同义词的概率越大。例如，对于《三体》小说中的两个词“海王星”和“天王星”，在《三体》语义共现网络中，“海王星”和“天王星”的邻居节点相似度很高。

<img src="https://github.com/tigerchen52/synonym_detection/blob/master/input/img/%E8%AF%AD%E4%B9%89%E7%BD%91%E7%BB%9C.png"  />
代码示例

```python
python synonym_detect -corpus_path  ../input/三体.txt -input_word_path ../temp/input_word.txt -process_number 2 -if_use_sn_model True
```
输出 :
```
5	阵列	阵列|矩形|队列|星体|量子
9	天空	天空|中|夜空|太阳|消失
4	海军	海军|航空兵|服役|空军|失败主义
10	建造	建造|制造|加速器|飞船|太阳系
3	椅子	椅子|桌子|坐下|沙发|台球桌
1	海王星	海王星|天王星|土星|卫星|群落
7	程心	程心|AA|中|罗辑|说
8	火焰	火焰|光芒|光点|推进器|雪峰
2	女孩	女孩|接待|冲何|请云|女士
6	变化	变化|发生|意味着|恢复|中
```
###  4. levenshtein距离
计算编辑距离发现同义词
代码示例

```python
python synonym_detect -corpus_path  ../input/三体.txt -input_word_path ../temp/input_word.txt -process_number 2 -if_use_leven_model True
```
输出 :
```
1	海王星	海王星|冥王星|天王星|星|王
7	程心	程心|请程心|带程心|连程心|从程心
6	变化	变化|变化很大|动态变化|发生变化|化
3	椅子	椅子|子|筐子|村子|棒子
2	女孩	女孩|女孩儿|女孩子|小女孩|女
10	建造	建造|建造成|造|建|建到
5	阵列	阵列|列|阵|历列|列为
9	天空	天空|海阔天空|空|天|天马行空
8	火焰	火焰|火|焰|火星|野火
4	海军	海军|于海军|陆海空军|海|海军军官
```
###  5. DPE模型
undo



