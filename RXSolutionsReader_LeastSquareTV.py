#!/usr/bin/env python
# coding: utf-8

#   Copyright 2025 UKRI-STFC

#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
# Authors:
# Franck Vidal (URKI-STFC)


# RXSolutionsReader Laminography Demo

## Data format: RX Solutions

# The data is in the format used by devices made by [RX Solutions](https://www.rx-solutions.com/en). The projections are saved in TIFF files. They are flatfield corrected using 16-bit unsigned integers. Metadata is saved in two different files, an XML file that can be used with orbital geometries, and a CSV file that can be used with flexible geometries.

## CIL Version

# This notebook was developed using CIL v25.0.0

## Dataset
# The data is available from Zenodo: https://doi.org/10.5281/zenodo.??????

# It is a laminography dataset of ???. 
# It was acquired with the ???? platform developed by [RX Solutions](https://www.rx-solutions.com/en) for the [MATEIS Laboratory](https://mateis.insa-lyon.fr/en) of [INSA-Lyon](https://www.insa-lyon.fr/en/).

# Update this filepath to where you have saved the dataset:


# data_path = "/DATA/CT/2025/DTHE"
# number_of_slices_to_reconstruct = 500 # Use 0 to compute it automatically
# pixel_pitch_in_mm = (0.15,0.15)
# scaling_factor = 3
# first_angle=360
# last_angle=0

# data_path = "/DATA/CT/2025/RX_Solutions/suzanne_circular"
# number_of_slices_to_reconstruct = 0 # Use 0 to compute it automatically
# pixel_pitch_in_mm = (0.5,0.5)
# scaling_factor = 3
# first_angle=0
# last_angle=360

file_path = os.path.join(data_path, 'unireconstruction.xml')
# file_path = os.path.join(data_path, 'geometry.csv')


import numpy as np
import gc

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('pdf')

from cil.processors import TransmissionAbsorptionConverter, Slicer, CentreOfRotationCorrector
from cil.framework import ImageGeometry
from cil.plugins.astra import FBP
from cil.io.TIFF import TIFFWriter

from readers.RXSolutionsDataReader import RXSolutionsDataReader


# Loading Geometry

if scaling_factor == 1:
    roi = None
else:
    roi = {"axis_1": [None, None, scaling_factor], "axis_2": [None, None, scaling_factor]}

reader = RXSolutionsDataReader(file_path, pixel_pitch_in_mm=pixel_pitch_in_mm, first_angle=first_angle, last_angle=last_angle, last_angle_included=False, roi=roi)

acq_geom = reader.get_geometry()

print(acq_geom)


acq_data = reader.read()

# Pre-processing

# In[ ]:


data_exp = TransmissionAbsorptionConverter()(acq_data)


# In[ ]:


if acq_geom.geom_type != "CONE_FLEX":
    processor = CentreOfRotationCorrector.image_sharpness("centre", "tigre")
    processor.set_input(data_exp)
    data_corr = processor.get_output()
else:
    data_corr = data_exp

# Prepare the data for Astra
data_corr.reorder(order='astra')


# In[ ]:


if acq_geom.geom_type != "CONE_FLEX":
    image_geometry = data_corr.geometry.get_ImageGeometry()

    image_geometry.voxel_size_x = min(image_geometry.voxel_size_x, image_geometry.voxel_size_y, image_geometry.voxel_size_z)
    image_geometry.voxel_size_y = image_geometry.voxel_size_x
    image_geometry.voxel_size_z = image_geometry.voxel_size_x
else:
    # Use the system magnification to compute the voxel size
    mag = data_corr.geometry.magnification
    mean_mag = np.mean(mag)
    print("Mean magnification: ", mean_mag)

    voxel_size_xy = data_corr.geometry.config.panel.pixel_size[0] / mean_mag
    voxel_size_z = data_corr.geometry.config.panel.pixel_size[1] / mean_mag

    # Create an image geometry
    num_voxel_xy = int(np.ceil(data_corr.geometry.config.panel.num_pixels[0]))
    num_voxel_z = int(np.ceil(data_corr.geometry.config.panel.num_pixels[1]))

    image_geometry = ImageGeometry(num_voxel_xy, num_voxel_xy, num_voxel_z, voxel_size_xy, voxel_size_xy, voxel_size_z)

if number_of_slices_to_reconstruct > 0:
    image_geometry.voxel_num_z = number_of_slices_to_reconstruct // scaling_factor

print(image_geometry)


# Using a FDK for the reconstruction

# In[ ]:


# Reconstruct using FDK
# Instantiate the reconsruction algorithm
fdk = FBP(image_geometry, data_corr.geometry)
fdk.set_input(data_corr)

# Perform the actual CT reconstruction
FDK_recon = fdk.get_output()


## Release memory

# In[ ]:


del data_exp
del acq_data
del reader

gc.collect();


## Save the reconstruction as a stack of TIFF files

writer = TIFFWriter(FDK_recon, os.path.join(data_path, "FDK-recon/slice"))
writer.write()


# Using TV regularised least squares solved with FISTA for the reconstruction

from cil.plugins.astra import ProjectionOperator
from cil.optimisation.functions import LeastSquares
from cil.plugins.ccpi_regularisation.functions import FGP_TV
from cil.optimisation.algorithms import FISTA

projector = ProjectionOperator(image_geometry, data_corr.geometry)
LS = LeastSquares(A=projector, b=data_corr)

alpha = 0.05
TV = FGP_TV(alpha=alpha, nonnegativity=True, device='gpu')
update_objective_interval = 10
number_of_iterations_per_loop = 25
fista_TV = FISTA(initial=FDK_recon, f=LS, g=TV, update_objective_interval=update_objective_interval)


# In[ ]:


fix_range = (FDK_recon.min(), FDK_recon.max())

for i in range(4):
    fista_TV.run(number_of_iterations_per_loop,verbose=1)

# Plot the evolution of the objective function
plt.figure()
plt.plot(np.linspace(0, (len(fista_TV.objective) - 1) * update_objective_interval, len(fista_TV.objective)), fista_TV.objective)
plt.savefig(os.path.join(data_path, 'fista-tv-objectives.pdf'))
plt.show()

TV_recon = fista_TV.solution


del fista_TV
del TV

gc.collect();

## Save the reconstruction as a stack of TIFF files

writer = TIFFWriter(TV_recon, os.path.join(data_path, "TV-recon/slice"))
writer.write()
