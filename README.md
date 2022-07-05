# CPGNet

参照GPGNet的论文：《CPGNet: Cascade Point-Grid Fusion Network for Real-Time LiDAR Semantic Segmentation》，2022年4月

复现了其各个网络层模块，其中一些细节可能与官方代码不同，

比如点云升维那里，论文中一开始说的是升到9维，但最后的相对角度没说是基于点云中心还是网格中心，我这里实现的是更为复杂的基于网格中心，后来跟作者聊了一下，他的代码是基于点云中心，而且维数减到7维了。。。

还有attention那里，我写的是concat之后，conv输入2c输出c，然后sigmoid之后再取最大降到1，也可以直接在conv那里输入2c，输出1。

2DFCN我在单个GTX 1080Ti 12GB显卡上测试了(1, 64, 64, 2048)RV的默认输入尺寸，平均耗时11ms，对于BEV(1, 64, 600, 600)没有测试，其尺寸差不多是RV的3倍，速度应该是30ms左右，加上其他的层，接近作者论文中的时间(43ms)。
