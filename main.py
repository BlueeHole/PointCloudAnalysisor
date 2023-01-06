
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import copy
import math
import seaborn as sns


def gen_standard_model(size, visual=False):
    mesh = o3d.io.read_triangle_mesh('suv.obj')
    # o3d.visualization.draw_geometries([textured_mesh])

    pcd = mesh.sample_points_uniformly(number_of_points=size)
    # pcd = mesh.sample_points_poisson_disk(number_of_points=50000, pcl=pcd)
    # print(len(pcd.points))
    # pcd = o3d.geometry.PointCloud.voxel_down_sample(pcd, voxel_size=0.1)
    if visual:
        o3d.visualization.draw_geometries([pcd])
    return pcd


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp],
                                      zoom=0.4459,
                                      front=[0.9288, -0.2951, -0.2242],
                                      lookat=[1.6784, 2.0612, 1.4451],
                                      up=[-0.3402, -0.9189, -0.1996])


def registration(source, target, visualize=False):
    print("Apply point-to-plane ICP")
    source.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.08, max_nn=30))
    target.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.08, max_nn=30))
    reg_p2l = o3d.pipelines.registration.registration_icp(
        source, target, 0.2, np.identity(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    print(reg_p2l)
    # print("Transformation is:")
    # print(reg_p2l.transformation)
    if visualize:
        draw_registration_result(source, target, reg_p2l.transformation)

    return reg_p2l.transformation


def my_hist(x):
    plt.rcParams["font.sans-serif"] = 'SimHei'
    plt.rcParams['axes.unicode_minus'] = False
    n, bins, patches = plt.hist(x, bins=50, density=True,
                                facecolor='#2ab0ff', edgecolor='#e0e0e0',
                                linewidth=0.5, alpha=0.7)
    for i in range(len(patches)):
        patches[i].set_facecolor(plt.cm.viridis(n[i] / max(n)))
    plt.title('Point-to-kNN-Point Distance', fontsize=12)
    plt.xlabel('distance(m)', fontsize=10)
    plt.ylabel('probability density', fontsize=10)
    plt.show()


def my_density_hist(x):
    sns.displot(x, bins=50, kde=True)
    plt.show()


def testWidth(stand_model):
    my_model = o3d.io.read_point_cloud('car_model_50.pcd')
    R = stand_model.get_rotation_matrix_from_xyz((0, 0, 0))
    # R = stand_model.get_rotation_matrix_from_xyz((0, 0, np.pi))
    # R = stand_model.get_rotation_matrix_from_xyz((0, 0, np.pi/2))
    my_model = my_model.translate((0, 0, 0), relative=False)
    stand_model = stand_model.translate((0, 0, 0), relative=False)
    stand_model.rotate(R, center=stand_model.get_center())
    registration(stand_model, my_model, True)
    # 就是用KNN法计算距离的
    # https://github.com/isl-org/Open3D/blob/master/cpp/open3d/geometry/PointCloud.cpp
    dist = my_model.compute_point_cloud_distance(stand_model)
    dist = np.asarray(dist)
    print(dist.mean())
    # ind = np.where(dist > 0.2)[0]
    # pcd3 = my_model.select_by_index(ind)
    # o3d.visualization.draw_geometries([pcd3], window_name="计算点云距离",
    #                                   width=800,  # 窗口宽度
    #                                   height=600)  # 窗口高度
    my_hist(dist)
    plt.show()


if __name__ == '__main__':
    print('Point Cloud Processor')
    '''
    Car Pose
    -6.63132, 1.85461, 0,
    0, 0, -1.57
    '''
    size = 467575
    stand_model = gen_standard_model(size, visual=False)
    stand_model.scale(0.06, center=stand_model.get_center())
    # o3d.io.write_point_cloud('stand_model.pcd', stand_model, write_ascii=True)

    my_model = o3d.io.read_point_cloud('my_model_transed.pcd')
    # R = stand_model.get_rotation_matrix_from_xyz((0, 0, -np.pi/2))
    # my_model = o3d.io.read_point_cloud('car_transed.pcd')
    # R = stand_model.get_rotation_matrix_from_xyz((np.pi / 2, -np.pi / 2, 0))
    # down_model = my_model.voxel_down_sample(voxel_size=0.01)
    down_model = my_model.uniform_down_sample(every_k_points=5)

    # 用静止帧验证宽度问题，发现还是宽度不一致，考虑是雷达的问题，不是GICP的问题
    testWidth(stand_model)

    # print(down_model.get_center())
    # print(stand_model.get_center())
    # # 配合my_model_transed的代码
    # down_model = down_model.translate(down_model.get_center(), relative=False)
    # stand_model = stand_model.translate((0, 0, 0), relative=False)
    # stand_model.rotate(R, center=stand_model.get_center())
    #
    # viewer = o3d.visualization.Visualizer()
    # viewer.create_window(width=1000, height=800)
    # viewer.add_geometry(stand_model)
    # viewer.add_geometry(down_model)
    # # viewer.add_geometry(a)
    # opt = viewer.get_render_option()
    # opt.show_coordinate_frame = True
    # # viewer.create_window(width=800, height=600)
    # viewer.run()
    # viewer.destroy_window()

    # plt.plot(dist)
    # plt.show()
    # my_hist(dist)