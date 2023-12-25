from plyfile import PlyData
import numpy as np
import torch
from diff_gaussian_renderer import render_gaussians, OptixState

ply_path = 'test/example.ply'

ply_data = PlyData.read(ply_path)
# Assuming 'ply_data' is already read from a PLY file and contains elements
first_element = ply_data.elements[0]
properties_lists = {prop.name: np.array(first_element.data[prop.name]) for prop in first_element.properties}
print(properties_lists.keys())
x = properties_lists['x']
y = properties_lists['y']
z = properties_lists['z']

opacity = properties_lists['opacity']

scale_x = properties_lists['scale_0']
scale_y = properties_lists['scale_1']
scale_z = properties_lists['scale_2']

a = properties_lists['rot_0']
b = properties_lists['rot_1']
c = properties_lists['rot_2']
d = properties_lists['rot_3']

means = np.column_stack((x, y, z))
scales = np.column_stack((scale_x, scale_y, scale_z))
rotations = np.column_stack((a, b, c, d))

means = torch.tensor([0,0,0])
scales = torch.tensor([0.5,0.5,0.5])
rotations = torch.tensor([0,0,0,1])
# res = torch.cuda.mem_get_info()
# print(res[0], res[1])
means = torch.from_numpy(means).to('cuda')
scales = torch.from_numpy(scales).to('cuda')
rotations = torch.from_numpy(rotations).to('cuda')
state = OptixState(2560, 1440)

# res = torch.cuda.mem_get_info()
# print(res[0], res[1])