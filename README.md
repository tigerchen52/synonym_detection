同义词挖掘
========



**同义词挖掘方法**：</br>

（1）百度百科同义词 </br>
（2）word2vector </br>
（3）语义共现网络的节点相似度 </br>
（4）levenshtein距离 </br>
（5）DPE模型（未完成）[【参考文献】](https://arxiv.org/pdf/1706.08186.pdf) </br>

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

