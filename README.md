# CPGNet

参照GPGNet的论文：《CPGNet: Cascade Point-Grid Fusion Network for Real-Time LiDAR Semantic Segmentation》，2022年4月

复现了其各个网络层模块，其中一些细节可能与官方代码不同，

比如点云升维那里，论文中一开始说的是升到9维，但最后的相对角度没说是基于点云中心还是网格中心，我这里实现的是更为复杂的基于网格中心，后来跟作者聊了一下，他的代码是基于点云中心，而且维数减到7维了。。。
