import numpy as np
import random
from scipy.spatial.transform import Rotation as R


# BEV类，点云转鸟瞰图，注意，俯视图中，点云的x坐标轴向前，跟图像的y轴相反，点云的y坐标轴向左，跟图像的x轴相反
# 为统一说明，投影前的xy均按照点云的xy坐标轴方向，投影后的xy均按照图像的xy坐标轴方向
class BevMap:
    """Class that contains LaserScan with x,y,z,r"""
    EXTENSIONS_SCAN = ['.bin']  # 所支持的点云后缀，暂时只支持.bin格式

    def __init__(self, project=False, proj_H=600, proj_W=600, xrange=(-50, 50), yrange=(-50, 50), zrange=(-2, 1.5),
                 DA=False, flip_sign=False, rot=False, drop_points=False):
        self.project = project  # 是否进行投影转换
        self.xrange = xrange  # 点云roi的前后范围
        self.yrange = yrange  # 点云roi的左右范围
        self.zrange = zrange  # 截取的高度范围
        self.proj_H = proj_H  # 投影后的图像高度
        self.proj_W = proj_W  # 投影后的图像宽度
        self.res = (xrange[1] - xrange[0]) / proj_H  # 分辨率，即每个像素代表几米*几米，默认1/6米

        # 点云增强
        self.DA = DA  # 随机抖动
        self.flip_sign = flip_sign  # y轴翻转(即点云左右翻转)
        self.rot = rot  # 随机旋转
        self.drop_points = drop_points  # 随机删点

        self.reset()  # 初始化

    def reset(self):
        """ Reset scan members. """
        self.points = np.zeros((0, 3), dtype=np.float32)  # 原始点坐标[m, 3]: x, y, z
        self.remissions = np.zeros((0, 1), dtype=np.float32)  # 原始反射率[m ,1]: remission
        # self.expd_points = np.zeros((0, 9), dtype=np.float32)  # 扩充后的点[m, 9]: x,y,z,i,r,xoff,yoff,theta,fi

        # projected range image - [H,W] range (-1 is no data)
        self.proj_bev = np.full((self.proj_H, self.proj_W, 9), -1,
                                dtype=np.float32)

        # unprojected range (list of depths for each point)
        self.unproj_bev = np.zeros((0, 1), dtype=np.float32)

        # unprojected z (list of heights for each point)
        self.unproj_z = np.zeros((0, 1), dtype=np.float32)

        # unprojected range (list of depths for each point)
        self.unproj_range = np.zeros((0, 1), dtype=np.float32)

        # projected point cloud xyz - [H,W,3] xyz coord (-1 is no data)
        self.proj_xyz = np.full((self.proj_H, self.proj_W, 3), -1,
                                dtype=np.float32)

        # projected remission - [H,W] intensity (-1 is no data)
        self.proj_remission = np.full((self.proj_H, self.proj_W), -1,
                                      dtype=np.float32)

        # projected z - [H,W] intensity (-1 is no data)
        self.proj_height = np.full((self.proj_H, self.proj_W), -1,
                                   dtype=np.float32)

        # projected depth - [H,W] intensity (-1 is no data)
        self.proj_depth = np.full((self.proj_H, self.proj_W), -1,
                                  dtype=np.float32)

        # projected index (for each pixel, what I am in the pointcloud)
        # [H,W] index (-1 is no data)
        self.proj_idx = np.full((self.proj_H, self.proj_W), -1,
                                dtype=np.int32)

        # for each point, where it is in the bev image
        # 针对原点云，每个点对应的投影索引
        self.proj_x = np.zeros((0, 1), dtype=np.int32)  # [m, 1]: x
        self.proj_y = np.zeros((0, 1), dtype=np.int32)  # [m, 1]: y
        # 针对ROI区域内的点云，每个点对应的索引
        self.px_cutted = np.zeros((0, 1), dtype=np.int32)  # [m, 1]: x
        self.py_cutted = np.zeros((0, 1), dtype=np.int32)  # [m, 1]: y

        # mask containing for each pixel, if it contains a point or not
        self.proj_mask = np.zeros((self.proj_H, self.proj_W),
                                  dtype=np.int32)  # [H,W] mask

    def size(self):
        """ Return the size of the point cloud. """
        return self.points.shape[0]

    def __len__(self):
        return self.size()

    def open_scan(self, filename):
        """ Open raw scan and fill in attributes
        """
        # reset just in case there was an open structure
        self.reset()

        # check filename is string
        if not isinstance(filename, str):
            raise TypeError("Filename should be string type, "
                            "but was {type}".format(type=str(type(filename))))

        # check extension is a laserscan
        if not any(filename.endswith(ext) for ext in self.EXTENSIONS_SCAN):
            raise RuntimeError("Filename extension is not valid scan file.")

        # if all goes well, open pointcloud
        scan = np.fromfile(filename, dtype=np.float32).reshape((-1, 4))

        # put in attribute
        points = scan[:, 0:3]  # get xyz
        remissions = scan[:, 3]  # get remission

        # 随机删点，我就不删了
        if self.drop_points is not False:
            self.points_to_drop = np.random.randint(0, len(points) - 1, int(len(points) * self.drop_points))
            points = np.delete(points, self.points_to_drop, axis=0)
            remissions = np.delete(remissions, self.points_to_drop)

        self.set_points(scan, points, remissions)

    def set_points(self, scan, points, remissions=None):
        """ Set scan attributes (instead of opening from file)
        """
        # reset just in case there was an open structure
        self.reset()
        # 初始化扩充点成员变量，考虑后四维的值域，设置一个比较特殊的初值
        self.expd_points = np.full((points.shape[0], 9), 8, dtype=np.float32)

        # check scan makes sense
        if not isinstance(points, np.ndarray):
            raise TypeError("Scan should be numpy array")

        # check remission makes sense
        if remissions is not None and not isinstance(remissions, np.ndarray):
            raise TypeError("Remissions should be numpy array")

        # put in attribute
        self.points = points  # get
        self.expd_points[:, 0:3] = points  # 扩充点的前三维是xyz
        # 点云的y轴是否翻转(即左右镜像)
        if self.flip_sign:
            self.points[:, 1] = -self.points[:, 1]
        # xyz坐标随机抖动，也就是模拟高斯噪声
        if self.DA:
            jitter_x = random.uniform(-5, 5)
            jitter_y = random.uniform(-3, 3)
            jitter_z = random.uniform(-1, 0)
            self.points[:, 0] += jitter_x
            self.points[:, 1] += jitter_y
            self.points[:, 2] += jitter_z
        # 随机旋转
        if self.rot:
            self.points = self.points @ R.random(random_state=1234).as_dcm().T
        # 反射率那一维赋值，如果没有反射率，默认赋0
        if remissions is not None:
            self.remissions = remissions  # get remission
            # if self.DA:
            #    self.remissions = self.remissions[::-1].copy()
        else:
            self.remissions = np.zeros((points.shape[0]), dtype=np.float32)
        self.expd_points[:, 3] = self.remissions  # 扩充点的第四维是反射率
        # if projection is wanted, then do it and fill in the structure
        if self.project:
            self.do_bev_projection()

    def do_bev_point_expand(self):
        '''
            点云channels维扩充维数，输入点为(x,y,z,i)，输出点为(x,y,z,i,r,xoff,yoff,pitch,yaw)
            r为range，即距离，
            xoff和yoff是相对于网格中心的偏移，网格即投影后的像素，
            pitch和yaw也是相对于网格中心的垂直角和水平角的偏移，论文中使用的maxpool，即使用一个网格中最大的数值代表整个网格

            以论文中的bev投影为例
            xoff和yoff的值域均为(-1/12, 1/12)，pitch的值域为(-pi/2, 0]，yaw的值域为[-pi, pi]

            点云的索引总共变化了3次
            以000000.bin为例，原点云数量是125635，经过ROI之后为124392，然后还要按z轴升序排序，
            最后proj投影，同一网格中删掉重复点，那只会剩下4万多点
            但是对于投影视图之外的点，扩充的后4维是直接赋默认值，还是依然计算真实值，论文中并没有说明
        '''
        depth = np.linalg.norm(self.points, ord=2, axis=1)  # 获得深度，即距离
        self.expd_points[:, 4] = depth
        # get scan components
        scan_x = self.points[:, 0]
        scan_y = self.points[:, 1]
        scan_z = self.points[:, 2]

        # 注意，俯视图中点云的x轴向前，与图像的y轴是相反的，点云的y轴向左，与图像的x轴是相反的
        bev_x = (-scan_y / self.res)
        bev_y = (-scan_x / self.res)
        bev_x_indx = (np.floor(bev_x)).astype(np.int32)
        bev_y_indx = (np.floor(bev_y)).astype(np.int32)

        # 调整坐标原点，点云默认的原点在中心，而图像的原点默认在左上角，其实就是x+1/2W，y+1/2H
        bev_x -= int(np.floor(self.yrange[0]) / self.res)
        bev_y += int(np.ceil(self.xrange[1]) / self.res)
        bev_x_indx -= int(np.floor(self.yrange[0]) / self.res)
        bev_y_indx += int(np.ceil(self.xrange[1] / self.res))

        # 计算xy相对于网格中心点的偏移，这里的偏移我用了原始的距离，如果要用调整分辨率后的距离，把后面的*self.res去掉
        # TODO: 其实这里有重复运算，相当于再计算了一遍每个点投影后的索引，后续可以考虑优化，按照计算zoff的方式
        xoff = abs(bev_x - (bev_x_indx + 0.5)) * self.res
        yoff = abs(bev_y - (bev_y_indx + 0.5)) * self.res
        self.expd_points[:, 5] = xoff
        self.expd_points[:, 6] = yoff

        # 排序之前，要把原始点云与投影后的xy坐标一一对应，后续反投影要用到，这里很重要
        # 注意，这里的xy下标可能有[0, 599]以外的数，因为为了保证与原点云一一对应，BEV投影之外的点并没有删除
        self.proj_x = np.copy(bev_x_indx)
        self.proj_y = np.copy(bev_y_indx)

        # BEV设定了原始点云视图的范围(-50m, 50m)，故有部分点落到视图之外了，要删除
        x_filt = np.logical_and(scan_x > self.xrange[0], scan_x < self.xrange[1])
        y_filt = np.logical_and(scan_y > self.yrange[0], scan_y < self.yrange[1])
        filter = np.logical_and(x_filt, y_filt)
        indices = np.argwhere(filter).flatten()

        self.px_cutted = self.proj_x[indices]
        self.py_cutted = self.proj_y[indices]
        scan_z_roi = scan_z[indices]
        bev_x_indx = bev_x_indx[indices]
        bev_y_indx = bev_y_indx[indices]
        xoff_roi = xoff[indices]
        yoff_roi = yoff[indices]
        # TODO: 点云截取完视图之后的idx是否要存一份？

        # 按照z轴升序排序，因为后面的赋值，如果同一像素有两个高度值，后赋的值会覆盖掉先赋的值，
        # 如果按作者取最大的原则，则就应该升序排序
        order = np.argsort(scan_z_roi)
        scan_z_ord = scan_z_roi[order]
        bev_x_indx = bev_x_indx[order]
        bev_y_indx = bev_y_indx[order]

        # 先把z投影到对应视图上，得到每个网格中的最大z值，再依照索引，将最大z值反投影回来，
        # 这样就得到了每个点所对应的网格中心的z值(最大z值)
        # 要注意，反投影的索引要与原点保持一致
        self.proj_height[bev_y_indx, bev_x_indx] = scan_z_ord
        self.unproj_height = self.proj_height[self.py_cutted, self.px_cutted]
        # 计算每个点相对于各自网格中心的z偏移和相对距离，然后计算相对垂直角和水平角
        # 其实可以预见，垂直角肯定<=0，因为中心点的z值最大嘛
        zoff = scan_z_roi - self.unproj_height
        roff = np.sqrt((xoff_roi ** 2 + yoff_roi ** 2 + zoff ** 2))
        pitch = np.arcsin(zoff / roff)
        # 水平角这里对于落在BEV视图之外的点，我没有用默认值，
        # 而是依然计算了这些点与离他们最近的网格中心的xy偏移值，然后算得水平角
        # 当然，也可以对这些点忽视，直接赋默认值，具体做法跟俯仰角一样
        yaw = np.arctan2(yoff, xoff)
        self.expd_points[indices, 7] = pitch
        self.expd_points[:, 8] = yaw

    def do_bev_projection(self):
        """
            将点云投影至鸟瞰图，需要的参数是点云ROI的范围(上下左右)，分辨率(或者投影后图像的尺寸)，高度(即z轴)截取的范围
        """
        # get depth of all points
        depth = np.linalg.norm(self.points, 2, axis=1)
        # get scan components
        scan_x = self.points[:, 0]
        scan_y = self.points[:, 1]
        scan_z = self.points[:, 2]
        # 获得区域内的点，注意俯视图中点云的坐标轴，x轴向前，y轴向左
        x_filt = np.logical_and(scan_x > self.xrange[0], scan_x < self.xrange[1])
        y_filt = np.logical_and(scan_y > self.yrange[0], scan_y < self.yrange[1])
        filter = np.logical_and(x_filt, y_filt)
        indices = np.argwhere(filter).flatten()
        scan_x = scan_x[indices]
        scan_y = scan_y[indices]
        scan_z = scan_z[indices]
        # 注意，俯视图中点云的x轴向前，与图像的y轴是相反的，点云的y轴向左，与图像的x轴是相反的
        proj_x = (-scan_y / self.res).astype(np.int32)
        proj_y = (-scan_x / self.res).astype(np.int32)
        # 调整坐标原点，点云默认的原点在中心，而图像的原点默认在左上角
        proj_x -= int(np.floor(self.yrange[0]) / self.res)
        proj_y += int(np.ceil(self.xrange[1] / self.res))
        self.proj_x = np.copy(proj_x)
        self.proj_y = np.copy(proj_y)
        # print(proj_x.min(), proj_x.max(), proj_y.min(), proj_y.max())

        # 把像素值归一化到(0, 255)
        # z_value = ((z_value - self.zrange[0]) / float(self.zrange[1] - self.zrange[0]) * 255).astype(np.int32)

        # copy of depth in original order
        # self.unproj_range = np.copy(depth)

        # order in decreasing depth
        # indices = np.arange(depth.shape[0])
        # order = np.argsort(depth)[::-1]
        # depth = depth[order]
        # indices = indices[order]
        # points = self.points[order]
        # z_value = scan_z[order]
        # remission = self.remissions[order]
        # proj_y = proj_y[order]
        # proj_x = proj_x[order]

        # 填充z坐标，这里的z坐标也可以做一下归一化操作
        # TODO:论文中貌似填充的是z轴的最大值，我觉得不妥，我改成了填充一个高度范围内的值，类似于点云的水平切片
        # z_value = np.clip(a=scan_z, a_min=self.zrange[0], a_max=self.zrange[1])

        # assing to images
        # 暂时按照LaserScan把各个图都赋值，具体来说depth和z_value是肯定用到的，其他不知道
        # self.proj_bev[proj_y, proj_x] = depth  # bev
        # self.proj_xyz[proj_y, proj_x] = self.points
        # self.proj_height[proj_y, proj_x] = z_value
        # self.proj_remission[proj_y, proj_x] = self.remission
        # self.proj_idx[proj_y, proj_x] = indices
        # self.proj_mask = (self.proj_idx > 0).astype(np.int32)


if __name__ == "__main__":
    print("\n", "-" * 50, " start ", "-" * 50, "\n")

    lidar_path = "/home/cidi/CPGNet_Plus/train/tasks/semantic/test_data/skitti/velodyne/000000.bin"  # 125635
    bev = BevMap(project=False)
    bev.open_scan(lidar_path)
    bev.do_bev_point_expand()

    # bev.expd_points的shape为(n, 9)，搭配上label文件构成Dataset，然后再封装成Dataloader，送到第一个MLP层


    print(bev.expd_points)
    print(bev.expd_points.shape)

    print("\n", "-" * 51, " end ", "-" * 51, "\n")
