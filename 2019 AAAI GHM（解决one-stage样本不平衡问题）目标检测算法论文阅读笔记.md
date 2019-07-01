## 2019 AAAI GHM（解决one-stage样本不平衡问题）目标检测算法论文阅读笔记

张凯 [极市平台](javascript:void(0);) *昨天*

> 加入极市专业CV交流群，与**6000+来自腾讯，华为，百度，北大，清华，中科院**等名企名校视觉开发者互动交流！更有机会与**李开复老师**等大牛群内互动！
>
> 同时提供每月大咖直播分享、真实项目需求对接、干货资讯汇总，行业技术交流**。**点击文末“**阅读原文**”立刻申请入群~



本文转载自知乎专栏：目标检测

来源：https://zhuanlan.zhihu.com/p/54182158

已获作者授权，请勿二次转载





# 

# 背景



《Gradient Harmonized Single-stage Detector》是2019 AAAI的Oral paper，出自港中文。这篇论文半年前就出来了，原理也比较简单，但当时认为相比于RetinaNet，GHM只有0.8个点的提升，所以感觉没有尝试的价值。但是最近结合Libra RCNN那篇关于balanced loss的讨论，以及目前在工程上的实现结果来看，性能提升还是非常明显的（特别是很多数据集标注存在误标的情况），值得分享下。



论文链接：

https://arxiv.org/pdf/1811.05181.pdf



代码地址：

https://github.com/libuyu/GHM_Detection



另外该代码基于港中文发布的mmdetection开发，目前港中文很多代码都是基于mmdetection实现，资源也较多，感兴趣的可以安装下。



# **一、研究动机：**



该论文主要聚焦于one-stage方法中样本不平衡问题，包括正样本和负样本、难易样本的不平衡问题。对比RetinaNet（focal loss）通过alpha参数控制正负样本比例、gamma参数控制难易样本比例，但是RetinaNet存在两个主要的问题，1）两个超参需要调整，在COCO上虽然影响不大，但是在产品中的训练样本来看，影响很大；2）没有考虑难例中outliers的影响，使得模型较难收敛，或者导致性能下降。



![img](https://mmbiz.qpic.cn/mmbiz_jpg/gYUsOT36vfoQgZjuAIk73yibiaibqydVmK9xbmecOLjT3ofxvvciay3hxicRtBS3GwI6Kk1dlibCUFgwV1bNGvZ6OB0w/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)



可以看到上图最右边Focal loss代表的蓝色的线在梯度为1的时候（outliers）比重是很大的。



# **二、具体方法**



**1 问题定义**

首先是对之前问题的定义，即之前的交叉熵损失函数：

![img](data:image/gif;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQImWNgYGBgAAAABQABh6FO1AAAAABJRU5ErkJggg==)



梯度求导非常简单：

![img](data:image/gif;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQImWNgYGBgAAAABQABh6FO1AAAAABJRU5ErkJggg==)



进一步简写为：

![img](data:image/gif;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQImWNgYGBgAAAABQABh6FO1AAAAABJRU5ErkJggg==)



**2 梯度密度**

引入梯度密度的概念：

![img](data:image/gif;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQImWNgYGBgAAAABQABh6FO1AAAAABJRU5ErkJggg==)



并且，

![img](data:image/gif;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQImWNgYGBgAAAABQABh6FO1AAAAABJRU5ErkJggg==)



物理意义非常简单，就是在一个区间内梯度的数目。从而引入梯度密度均衡参数：

![img](data:image/gif;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQImWNgYGBgAAAABQABh6FO1AAAAABJRU5ErkJggg==)





物理意义就是，密度越大，该部分的权重会被降低。



**3 GHM-C函数**

主要是对分类一支的改进，将梯度密度均衡参数引入：

![img](data:image/gif;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQImWNgYGBgAAAABQABh6FO1AAAAABJRU5ErkJggg==)



**4 GHM-R函数**

首先是对smooth L1的改进（主要是d可以无限大），所以引入了提出了ASL1：

![img](data:image/gif;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQImWNgYGBgAAAABQABh6FO1AAAAABJRU5ErkJggg==)



其梯度为：

![img](data:image/gif;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQImWNgYGBgAAAABQABh6FO1AAAAABJRU5ErkJggg==)



然后在此基础上加入梯度密度均衡参数：

![img](data:image/gif;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQImWNgYGBgAAAABQABh6FO1AAAAABJRU5ErkJggg==)



# 三、实验结果



![img](data:image/gif;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQImWNgYGBgAAAABQABh6FO1AAAAABJRU5ErkJggg==)





首先是超参的影响，M代表了划分区间，越大的话，越接近密度，但是由于每次迭代样本的随机性，也就越不稳定。在COCO上，M=30最好。



![img](data:image/gif;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQImWNgYGBgAAAABQABh6FO1AAAAABJRU5ErkJggg==)



GHMC和FL在COCO上表现差不多。


![img](data:image/gif;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQImWNgYGBgAAAABQABh6FO1AAAAABJRU5ErkJggg==)



GHMR比SL提升了0.6个百分点。



![img](data:image/gif;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQImWNgYGBgAAAABQABh6FO1AAAAABJRU5ErkJggg==)



最好的结果，在不同的backbone上，均有0.8个点的提升。



# 四、总结分析



GHM的思想比FL更进一步地解决了样本不平衡的问题，虽然在COCO上提升不大，但是在某些比较脏的数据集上，表现非常好，值得尝试。

（完）







*延伸阅读

- [ILSVRC2016目标检测任务回顾（上）--图像目标检测（DET）](http://mp.weixin.qq.com/s?__biz=MzI5MDUyMDIxNA==&mid=2247483911&idx=1&sn=08f9843720f2637215c88983feaff7e1&chksm=ec1feffedb6866e8a5b96cad8b83e7eae4bdf788369838615a72ebd0586de8ab4fe6a9051203&scene=21#wechat_redirect)

- [ILSVRC2016目标检测任务回顾（下）--视频目标检测（VID）](http://mp.weixin.qq.com/s?__biz=MzI5MDUyMDIxNA==&mid=2247483916&idx=1&sn=0257a52a4620297f696d2b1870c5b48a&chksm=ec1feff5db6866e3703f535fba6e58f67696c848bbd938368d8a738dbc7f106743c22bc8e70c&scene=21#wechat_redirect)

- [基于高性能检测器与表观特征的多目标跟踪（计算机视觉论文阅读笔记）](http://mp.weixin.qq.com/s?__biz=MzI5MDUyMDIxNA==&mid=2247483977&idx=1&sn=1c062825a6e2e24d044903abc78429b8&chksm=ec1fefb0db6866a6223539ed86e50e8d3b72dc10d2cac6a92e95d266c6da16ceb376d048e8d2&scene=21#wechat_redirect)

  





------

点击左下角**“阅读原文”，**即可申请加入极市**目标跟踪、目标检测、工业检测、人脸方向、视觉竞赛等技术交流群，**更有每月大咖直播分享、真实项目需求对接、干货资讯汇总，行业技术交流，一起来让思想之光照的更远吧~



![img](data:image/gif;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQImWNgYGBgAAAABQABh6FO1AAAAABJRU5ErkJggg==)

△长按关注极市平台



**觉得有用麻烦给个在看啦~**  **![img](data:image/gif;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQImWNgYGBgAAAABQABh6FO1AAAAABJRU5ErkJggg==)**

[阅读原文](https://mp.weixin.qq.com/s?__biz=MzI5MDUyMDIxNA==&mid=2247489401&idx=1&sn=4e819e8c1891fe287d65c8997e72d505&chksm=ec1ffa80db68739697af06cb592874fae5a5b6c2e8d35836d25e605ce982fb5c783f386d8b25&mpshare=1&scene=1&srcid=0701xOfHGscBwLlxxPZm4ZVr&key=90581f21d61583ccebf5671a741bbdeb2fad76f208c8ec3c7640e0a03de1ac6ec68b70fba1237374242b9c69c6425a4b24eebec41e09f0b5f6e3fd315ddbbe9ce0a322ffa17c5dee1a15bc9cb8ce44d9&ascene=1&uin=MjMzNDA2ODYyNQ%3D%3D&devicetype=Windows+10&version=62060833&lang=zh_CN&pass_ticket=OCAJEER4OwwriRfMF7Kv6joxIfS9N%2FzDtrEExbop1m8DXq4ger8RUp9307aHi2y7##)





![img](https://mp.weixin.qq.com/mp/qrcode?scene=10000004&size=102&__biz=MzI5MDUyMDIxNA==&mid=2247489401&idx=1&sn=4e819e8c1891fe287d65c8997e72d505&send_time=)

微信扫一扫
关注该公众号