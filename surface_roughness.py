# This program aims to calculate the surface roughness Ra and Rq.

import vtk
import numpy as np
from vtk.numpy_interface import dataset_adapter as dsa


Filename = '3d_1573_4.vtk'
# Use 'vtkStructuredPointsReader' to read 'structured_points' type data.
reader = vtk.vtkStructuredPointsReader()
reader.SetFileName(Filename)
reader.Update()

# Get the data
structured_points = reader.GetOutput()

# Get the number of points, which is 512 * 256 * 256
num_points = structured_points.GetNumberOfPoints()
print(f"Number of points: {num_points}")

# Get the scalar data of points
point_data = structured_points.GetPointData()
point_scalars = point_data.GetScalars()

# Use 'dsa.WrapDataObject' to wrap .VTK data
data = dsa.WrapDataObject(structured_points)

# Transform it to NumPy data
point_values = data.PointData[point_scalars.GetName()]

# Reshape it to a 3D tensor (512, 256, 256)
point_tensor = np.reshape(point_values, (256, 256, 512))

# Output the shape of tensor
print(f"Tensor shape: {point_tensor.shape}")

# The dimensions of point_tensor: (z=)256,(y=)256,(x=)512
# The y-direction in tensor is reverse with the y-direction in vtk file.
# But we just calculate the surface roughness here, so we ignore it.

surf_position = np.zeros((256,512))

for i_y in range(256):
    for i_x in range(512):
        for i_z in range(256):
            if point_tensor[i_z,i_y,i_x] == 0:
                surf_position[i_y,i_x] = i_z
                break

# Surface height (um)
surf_height = surf_position*0.5
surf_height = surf_height.astype(np.float32)
print('The shape of surface height matrix:',surf_height.shape)

# Surface average height (um)
ave_height = np.mean(surf_height)

print('The average height (um):',ave_height)

# Calculate the surface roughness
N = 256*512

delta_height = abs(surf_height-ave_height)
delta_height2 = (surf_height-ave_height)**2

height_total_1 = np.sum(delta_height)
print('The sum of delta_height',height_total_1)
height_total_2 = np.sum(delta_height2)
print('The sum of delta_height2',height_total_2)

Ra = height_total_1/N
print('Surface roughness Ra:',Ra)
Rq = height_total_2/N
print('Surface roughness Rq:',Rq)



