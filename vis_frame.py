from mayavi import mlab
import mayavi
mlab.options.offscreen = False
print("Set mlab.options.offscreen={}".format(mlab.options.offscreen))

import numpy as np

fov_voxels = []
with open('./fov_voxels.txt', 'r', encoding='utf-8') as file:  
    lines = [line.strip() for line in file]
    for line in lines:
        data_list = line.split(" ")
        fov_voxels.append([eval(data_list[0]), eval(data_list[1]), eval(data_list[2]), eval(data_list[3])])
file.close()
fov_voxels = np.array(fov_voxels)
voxel_size = [1.024, 1.024, 1.0]
figure = mlab.figure(size=(2560, 1440), bgcolor=(1, 1, 1))
# Draw occupied inside FOV voxels
voxel_size = sum(voxel_size) / 3
plt_plot_fov = mlab.points3d(
    fov_voxels[:, 0],
    fov_voxels[:, 1],
    fov_voxels[:, 2],
    fov_voxels[:, 3],
    colormap="viridis",
    scale_factor=0.95 * voxel_size,
    mode="cube",
    opacity=1.0,
    vmin=1,
    vmax=19, # 16
)

colors = np.array(
    [
        [255, 120,  50, 255],       # barrier              orange
        [255, 192, 203, 255],       # bicycle              pink
        [255, 255,   0, 255],       # bus                  yellow
        [  0, 150, 245, 255],       # car                  blue
        [  0, 255, 255, 255],       # construction_vehicle cyan
        [255, 127,   0, 255],       # motorcycle           dark orange
        [255,   0,   0, 255],       # pedestrian           red
        [255, 240, 150, 255],       # traffic_cone         light yellow
        [135,  60,   0, 255],       # trailer              brown
        [160,  32, 240, 255],       # truck                purple                
        [255,   0, 255, 255],       # driveable_surface    dark pink
        # [175,   0,  75, 255],       # other_flat           dark red
        [139, 137, 137, 255],
        [ 75,   0,  75, 255],       # sidewalk             dard purple
        [150, 240,  80, 255],       # terrain              light green          
        [230, 230, 250, 255],       # manmade              white
        [  0, 175,   0, 255],       # vegetation           green
        [  0, 255, 127, 255],       # ego car              dark cyan
        [255,  99,  71, 255],       # ego car
        [  0, 191, 255, 255]        # ego car
    ]
).astype(np.uint8)

plt_plot_fov.glyph.scale_mode = "scale_by_vector"
plt_plot_fov.module_manager.scalar_lut_manager.lut.table = colors

scene = figure.scene
scene.camera.position = [  0.75131739, -35.08337438,  16.71378558]
scene.camera.focal_point = [  0.75131739, -34.21734897,  16.21378558]
scene.camera.view_angle = 40.0
scene.camera.view_up = [0.0, 0.0, 1.0]
scene.camera.clipping_range = [0.01, 300.]
scene.camera.compute_view_plane_normal()
scene.render()

# scene.camera.position = [ 0.75131739,  0.78265103, 93.21378558]
# scene.camera.focal_point = [ 0.75131739,  0.78265103, 92.21378558]
# scene.camera.view_angle = 40.0
# scene.camera.view_up = [0., 1., 0.]
# scene.camera.clipping_range = [0.01, 400.]
# scene.camera.compute_view_plane_normal()
# scene.render()

mlab.savefig('result_trt_no_points.jpg')  # !!!
# mlab.show()