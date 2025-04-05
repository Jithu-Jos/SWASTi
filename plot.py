import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.gridspec as gridspec
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.image as mpimg
import imageio
import os
import sunpy
import pyPLUTO.pload as pp
from PIL import Image
from datetime import datetime, timedelta

logo_path = "./SWASTi_logo.png" 
logo = mpimg.imread(logo_path)


#2284
start_time_str = "2024-04-24 08:35:57.269"
time_cr = 2355773.3224

cr = 2284
model = 'FR12345'  # cone / FR
unit_time_seconds = 1.496e+13 / 2.500e+07 
tstart = 0
tstop = 7
num_files = 300
dt = 0.04
start_time = datetime.strptime(start_time_str, "%Y-%m-%d %H:%M:%S.%f") - timedelta(seconds=time_cr)
                
images = []
ticks = 15
title = 30
label = 15

unit_vel = 250.0
unit_rho = 10.0
unit_T = 7.517e+06/1.0e6
n_1AU = 67

if model == 'FR12345':
    inputdir = f"./Fri3d_12345/"
    outputdir = f"./Fri3d_12345/"
    outputgif = f"./Fri3d_12345.gif"
    titletext = f"Fri3D model CME, CR: {cr}"
elif model == 'FR145':
    inputdir = f"./Fri3d_145/"
    outputdir = f"./Fri3d_145/"
    outputgif = f"./Fri3d_145.gif"
    titletext = f"Fri3D model CME, CR: {cr}"
else:
    print("Error: Invalid model")

print(outputdir)
os.makedirs(outputdir, exist_ok=True)

for j in range(0, 50):
    D_mhd = pp.pload(j, inputdir, datatype='vtk')
    Vr_mhd = D_mhd.vx1 * unit_vel 
    rho_mhd = unit_rho * D_mhd.rho  
    T_mhd = unit_T * (D_mhd.prs / D_mhd.rho) * 1.0 

    Vr_rphi = Vr_mhd[:, 30, :]
    rho_rphi = rho_mhd[:, 30, :]
    Br_rphi = D_mhd.Bx1[:, 30, :]
    T_thetaphi = T_mhd[n_1AU, :, :]

    mhd_phi = np.linspace(0, 2 * np.pi, 181)
    mhd_r = np.linspace(0.1, 2.1, 150)

    phi, r = np.meshgrid(mhd_phi, mhd_r) 
    X1 = phi
    Y1 = r

    Vr_max = 800
    Vr_min = 280
    rho_max = 60
    rho_min = 10
  

    fig = plt.figure(figsize=(15, 8))
    plt.suptitle(titletext, fontsize=title, y=0.98)
    gs = gridspec.GridSpec(nrows=1, ncols=2, top=0.97, bottom=0.06, left=0.06, right=0.94, wspace=0.2)

    # First subplot for image1
    ax1 = fig.add_subplot(gs[0, 0], projection='polar')
    image1 = ax1.pcolormesh(X1, Y1, rho_rphi * (r)**2, vmin = rho_min, vmax = rho_max, shading='nearest', cmap='jet_r')
    ax1.set_theta_zero_location('E')
    ax1.scatter(0, 1, marker='o', s=120, fc='b', ec='white', label='Earth')
    ax1.scatter(0, 0, marker='o', s=120, fc='red', label='Sun')
    ax1.set_thetagrids([0, 60, 120, 180, 240, 300], fontsize=15)
    ax1.tick_params(pad=18)
    ax1.set_rgrids([1], labels=None, fontsize=0)
    ax1.set_xlabel("(a)", fontsize=15)
    ax1.minorticks_on()
    ax1.legend(loc='upper right', bbox_to_anchor=(1.1, 1))
    ax1.grid(alpha=0.6)
    cbar1 = fig.colorbar(image1, ax=ax1, orientation='vertical', shrink=0.7, pad=0.1, extend='both')
    cbar1.ax.tick_params(labelsize=ticks)
    cbar1.set_label("Scale Density", fontsize=label)

    # Second subplot for image2
    ax2 = fig.add_subplot(gs[0, 1], projection='polar')
    image2 = ax2.pcolormesh(X1, Y1, Vr_rphi, vmin=Vr_min, vmax=Vr_max, shading='auto', cmap='plasma')
    ax2.set_theta_zero_location('E')
    ax2.scatter(0, 1, marker='o', s=120, fc='b', ec='white', label='Earth')
    ax2.scatter(0, 0, marker='o', s=120, fc='red', label='Sun')
    ax2.set_thetagrids([0, 60, 120, 180, 240, 300], fontsize=15)
    ax2.tick_params(pad=18)
    ax2.set_rgrids([1], labels=None, fontsize=0)
    ax2.set_xlabel("(b)", fontsize=15)
    ax2.minorticks_on()
    ax2.legend(loc='upper right', bbox_to_anchor=(1.1, 1))
    ax2.grid(alpha=0.6)
    cbar2 = fig.colorbar(image2, ax=ax2, orientation='vertical', shrink=0.7, pad=0.1,extend='both')
    cbar2.ax.tick_params(labelsize=ticks)
    cbar2.set_label("Velocity [km/s]", fontsize=label)
    
    if j>=0:
	    time_increment = (j+125) * dt * unit_time_seconds
	    vtk_time = start_time + timedelta(seconds=time_increment)
	    vtk_time_ = vtk_time.strftime("%d-%m-%Y %H:%M:%S")
	    fig.text(0.5, 0.1, f"{vtk_time_}", ha='center', fontsize=25)
    # print(vtk_time_)
    
    imagebox = OffsetImage(logo, zoom=0.15)  
    ab = AnnotationBbox(imagebox, (0.00, -0.22), xycoords='axes fraction', frameon=False)
    ax1.add_artist(ab)
    
    plt.tight_layout()
    plt.savefig(os.path.join(outputdir, f"frame_{j}.png"), dpi=200)
    images.append(imageio.imread(os.path.join(outputdir, f"frame_{j}.png")))
    plt.close(fig)


# Create GIF
frames = [Image.open(os.path.join(outputdir, f"frame_{j}.png")) for j in range(1,50)]

frames[0].save(outputgif, save_all=True, append_images=frames[1:],  duration=200, loop=0)
               
print( f"gif saved as {outputgif}")

