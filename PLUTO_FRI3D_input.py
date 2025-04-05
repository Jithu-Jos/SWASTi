
import numpy as np
from ai import cs
from ai.fri3d.model import StaticFRi3D
from astropy import units as u
from matplotlib import pyplot as plt
from PIL import Image
#from mpl_toolkits.mplot3d import proj3d

#%% PLUTO setup

v_unit = 250.0                #km/s
l_unit = u.au.to(u.km, 1)   #in km
t_unit = l_unit/v_unit      #in sec

dt = t_unit*1.88e-4          #aprrox value in sec: has to be greater than the real g_dt
theta_res = 2.0               #in degree
phi_res = 2.0                 #in degree
#v_cme = 1000.0                #velocity of injecting CME in km/s


cme_phi = u.deg.to(u.rad, 9.0+180)
cme_theta = u.deg.to(u.rad, -7.0)  



hw_rad = u.deg.to(u.rad, 43.0)
hh_rad = u.deg.to(u.rad, 43.0)
v_t = 870    #v_cme/(1 + np.tan(hh_rad))


time = dt*np.arange(0, 240, 1)
radius = 0.081 + u.km.to(u.au, v_t*time)
cr = 2284
output_dir = f"Input/CME1"
input_dir = f"Input/CME1/"
output_dir_gif = f"{cr}/Input_gif/CME_trial_periodic_new/"
outputgif = f"{cr}/Input_gif/CME_trial_periodic_new/Input.gif"

for i in range (0, len(radius)):
    sfr = StaticFRi3D(toroidal_height=u.au.to(u.m, radius[i]),
        # poloidal_height=u.au.to(u.m, 0.4),
        latitude=cme_theta,
        longitude=cme_phi,
        half_width=hw_rad,
        half_height=hh_rad,
        pancaking=0.6,
        tilt = u.deg.to(u.rad, 0),
        chirality=-1.0,
        polarity=-1.0,
        flattening=0.5,
        flux=5e12,
        twist=2)
    
    _phi = np.linspace(-u.deg.to(u.rad, 90) + cme_phi, u.deg.to(u.rad, 90) + cme_phi, 181)
    _theta = np.linspace(-u.deg.to(u.rad, 60) + cme_theta, u.deg.to(u.rad, 60) + cme_theta, 121)
    _r = u.au.to(u.m, 0.1)
    
    r, theta, phi = np.meshgrid(_r, _theta, _phi)
    
    x, y, z = cs.sp2cart(r.flatten(), theta.flatten(), phi.flatten())
    
    b, v = sfr.data(x, y, z)
    b *= 1e9
    
    B_mag = np.sqrt(b[:,0]**2 + b[:,1]**2 + b[:,2]**2)
    
    m = np.logical_and(B_mag>0, radius[i]>0.08)
    
    vr = v_t * (v[m,0] + v[m,1]*np.cos(phi.flatten()[m]))

    print("\nSlice number : ", i)
    print('No. of points = "', len(x[m]),'" at', radius[i], 'AU')
    print('Vr = ', np.max(vr))                     #toroidal component
    #print('V_p = ', np.mean(v[m,1])*velocity)                     #poloidal component
    
    theta_pluto = u.deg.to(u.rad, 90) - theta.flatten()[m]
    phi_pluto = u.deg.to(u.rad, 180) + phi.flatten()[m]  
    # phi_pluto = u.deg.to(u.rad) + phi.flatten()[m]
    
#     phi_pluto = np.mod(phi_pluto, 2*np.pi)   # To make sure phi is in the range 0 to 2*pi
    phi_pluto = np.where(phi_pluto == 2*np.pi, 2*np.pi, np.mod(phi_pluto, 2*np.pi))

#     theta_pluto = np.mod(theta_pluto, np.pi) ##############
#     phi_pluto = np.where(phi_pluto > 2*np.pi,phi_pluto - 2*np.pi,phi_pluto)
#     phi_pluto = np.where(phi_pluto < 0,phi_pluto + 2*np.pi,phi_pluto)
    
    sin_t = np.sin(theta_pluto)
    sin_p = np.sin(phi_pluto)
    cos_t = np.cos(theta_pluto)
    cos_p = np.cos(phi_pluto)
    
    Br = b[m,0]*sin_t*cos_p + b[m,1]*sin_t*sin_p + b[m,2]*cos_t
    Bt = b[m,0]*cos_t*cos_p + b[m,1]*cos_t*sin_p - b[m,2]*sin_t
    Bp = -1*b[m,0]*sin_p + b[m,1]*cos_p
    Vr = vr
    
    fg1 = open(f"{output_dir}/FR_Br_"+str(i)+".txt", "w")
    fg2 = open(f"{output_dir}/FR_Bt_"+str(i)+".txt", "w")
    fg3 = open(f"{output_dir}/FR_Bp_"+str(i)+".txt", "w")
    fg4 = open(f"{output_dir}/FR_Vr_"+str(i)+".txt", "w")
    fg5 = open(f"{output_dir}/FR_theta_"+str(i)+".txt", "w")
    fg6 = open(f"{output_dir}/FR_phi_"+str(i)+".txt", "w")
    
    for j in range(0, len(Br)):
        
        fg1.write("%lf\n"%(Br[j]))
        fg2.write("%lf\n"%(Bt[j]))
        fg3.write("%lf\n"%(Bp[j]))
        fg4.write("%lf\n"%(Vr[j]))
        fg5.write("%lf\n"%(theta_pluto[j]))
        fg6.write("%lf\n"%(phi_pluto[j]))
        
    fg1.close()
    fg2.close()
    fg3.close()
    fg4.close()
    fg5.close()
    fg6.close()

images = []
num_frames = 240  # 
os.makedirs(output_dir, exist_ok=True)

for i in range(0,240):
    try:
        Br = np.loadtxt(f"{input_dir}FR_Br_{i}.txt")
        Bt = np.loadtxt(f"{input_dir}FR_Bt_{i}.txt")
        Bp = np.loadtxt(f"{input_dir}FR_Bp_{i}.txt")
        Vr = np.loadtxt(f"{input_dir}FR_Vr_{i}.txt")
        theta_pluto = np.loadtxt(f"{input_dir}FR_theta_{i}.txt")
        phi_pluto = np.loadtxt(f"{input_dir}FR_phi_{i}.txt")
        
        fig, axes = plt.subplots(2, 2, figsize=(10, 8), subplot_kw={'projection': 'polar'})
        fig.suptitle(f"Frame {i+1}")
        
        scatter_params = dict(cmap='viridis', s=10)
        
        valid_phi = (phi_pluto >= 0) & (phi_pluto <= 2 * np.pi)
        invalid_phi = ~valid_phi
        
        sc1 = axes[0, 0].scatter(phi_pluto[valid_phi], theta_pluto[valid_phi], c=Br[valid_phi], vmin=-60, vmax = 60, **scatter_params)
        axes[0, 0].scatter(phi_pluto[invalid_phi], theta_pluto[invalid_phi], color='r', s=10)
        axes[0, 0].set_title("Br")
        plt.colorbar(sc1, ax=axes[0, 0])
        
        sc2 = axes[0, 1].scatter(phi_pluto[valid_phi], theta_pluto[valid_phi], c=Bt[valid_phi], vmin=-60, vmax = 60, **scatter_params)
        axes[0, 1].scatter(phi_pluto[invalid_phi], theta_pluto[invalid_phi], color='r', s=10)
        axes[0, 1].set_title("Bt")
        plt.colorbar(sc2, ax=axes[0, 1])
        
        sc3 = axes[1, 0].scatter(phi_pluto[valid_phi], theta_pluto[valid_phi], c=Bp[valid_phi], vmin=-40, vmax = 40, **scatter_params)
        axes[1, 0].scatter(phi_pluto[invalid_phi], theta_pluto[invalid_phi], color='r', s=10)
        axes[1, 0].set_title("Bp")
        plt.colorbar(sc3, ax=axes[1, 0])
        
        sc4 = axes[1, 1].scatter(phi_pluto[valid_phi], theta_pluto[valid_phi], c=Vr[valid_phi], vmin=600, vmax = 1000, **scatter_params)
        axes[1, 1].scatter(phi_pluto[invalid_phi], theta_pluto[invalid_phi], color='r', s=10)
        axes[1, 1].set_title("Vr")
        plt.colorbar(sc4, ax=axes[1, 1])
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(f"{output_dir_gif}frame_input_{i+1}.png")
        images.append(f"{output_dir_gif}frame_input_{i+1}.png")
        plt.close()
        
        print(f"Saved frame_input_{i+1}.png")
    except Exception as e:
        print(f"Error processing frame {i}: {e}")

frames = [Image.open(f"{output_dir_gif}frame_input_{j}.png") for j in range(1,240)]

frames[0].save(outputgif, save_all=True, append_images=frames[1:],  duration=200, loop=0)
               
print( f"gif saved as {outputgif}")


