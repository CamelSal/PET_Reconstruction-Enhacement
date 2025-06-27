import pylab as plb
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import factorial
from skimage.transform import iradon_sart
from skimage.transform import iradon
from scipy.interpolate import interp1d

# Load column data from text files
def loadfile(fname,col):
    return np.loadtxt(fname,comments=['D','#'] ,usecols=(col,))

# Load projection data (every 18 degrees from 0 to 360)
data1 = []
t = 361
for i in range(0,t,18):
    x4 = loadfile("Data/Scan4_" + str(i) + "Deg.txt", 1)
    data1.append(x4)


data0= np.column_stack(data1) # Combine into sinogram

theta = np.arange(0,t,18) # Projection angles

# Load position axis (x) for projections and shift by +10 mm
x0 = loadfile("Data/Scan4_0Deg.txt",0)+ 10



# Two-peak Cauchy distribution (symmetric double peak)
def cauchy2d(x,x02,x01,s,y,a):
    return a/(np.pi*s*(1+((x-x02)/s)**2))+y+a/(np.pi*s*(1+((x-x01)/s)**2))

# Sinusoidal function used to fit the angular behavior of peak positions
def fun(x,a,p,y,c):
    return a*np.sin(x*p+y)+c

sim1 = []      # Simulated fitted curves
prm = []       # Parameters from fitting
pk1 = []       # Peak 1 positions
pk2 = []       # Peak 2 positions
sd = []        # Scale parameter
a = []         # Amplitude
yoff = []      # Offset

t= np.arange(10.0,-10.0,-0.05)


popt=[0,0,1.739,254,11985] # Initial guess for parameters [x02, x01, s, y, a]

for i in range(0,21):
    y= data0[:,i]
    popt,pcov = curve_fit(cauchy2d,x0,y,p0=popt,sigma=np.sqrt(y)) # Fit data
    f = cauchy2d(t,popt[0],popt[1],popt[2],popt[3],popt[4]) # Get fitted curve
    # Save results
    sim1.append(f)
    pk1.append(popt[0])
    pk2.append(popt[1])
    sd.append(popt[2])
    yoff.append(popt[3])
    a.append(popt[4])
    prm.append(popt)

# Plot fit optional
n= 5
plt.plot(x0,data0[:,n],'o')
plt.plot(t,sim1[n],'orange',linewidth=2)
plt.xlabel('Projection Axis (mm)')
plt.ylabel('Intensity (counts)')
plt.title("Cauchy Double Peak Curve Fit")
plt.legend(['Data Point','Cauchy Curve Fit'])
plt.show()

#  Organize peak position arrays for interpolation
l1= [pk2[0]]
l2= [pk1[0]]
peak1=l1+pk1[1:11]+pk2[11:21]
peak2=l2+pk2[1:11]+pk1[11:21]

peak1=np.array(peak1)

sim =np.column_stack(sim1) # Column stack all fitted projections

thetas= np.arange(0,360.1,0.1) # High-resolution angular domain

# Fit peak positions over angle using sinusoidal functions
psin1,pcov = curve_fit(fun,theta,peak1,p0=[-4,0.02,-1,0])
psin2,pcov = curve_fit(fun,theta,peak2,p0=[4,0.016,-1.1,0])

# Evaluate fitted functions over fine angle grid
csin1=fun(thetas,psin1[0],psin1[1],psin1[2],psin1[3])
csin2=fun(thetas,psin2[0],psin2[1],psin2[2],psin2[3])


#s = 1.739
tm1 = []

# Interpolate width and amplitude parameters over angle
fa = interp1d(theta,a,kind='cubic')
fs = interp1d(theta,sd,kind='cubic')


# Plot Parameters interpolations
fig_interp = plt.figure(figsize=(10, 14))  # Taller figure for stacked layout

# First plot: Sigma (width) interpolation
ax1 = plt.subplot2grid((3, 1), (0, 0))
ax1.plot(theta, sd, 'o', label='Scale')
ax1.plot(thetas, fs(thetas), color='orange', label='Interpolation')
ax1.set_ylabel("σ value")
ax1.set_xlabel("Theta (Deg)")
ax1.set_title("Scale Parameter Interpolation")
ax1.legend()

# Second plot: Amplitude interpolation
ax2 = plt.subplot2grid((3, 1), (1, 0))
ax2.plot(theta, a, 'o', label='Amplitude')
ax2.plot(thetas, fa(thetas), color='orange', label='Interpolation')
ax2.set_ylabel("a value")
ax2.set_xlabel("Theta (Deg)")
ax2.set_title("Amplitude Parameter Interpolation")
ax2.legend()

# Third plot: Peak positions interpolation
ax3 = plt.subplot2grid((3, 1), (2, 0))
ax3.plot(theta, peak1, 'o', label='Peak 1', color='red')
ax3.plot(theta, peak2, 'o', label='Peak 2', color='green')
ax3.plot(thetas, csin1, '--', color='orange', label='Fit Peak 1')
ax3.plot(thetas, csin2, '--', color='blue', label='Fit Peak 2')
ax3.set_ylabel("Position (mm)")
ax3.set_xlabel("Theta (Deg)")
ax3.set_title("Peak Position Interpolation")
ax3.legend()

plt.tight_layout()
plt.savefig('figures/parameter_interpolation.png')  # Optional: save the figure
plt.show()


# Generate enhance projection

for i in thetas:
    f1 = fun(i,psin1[0],psin1[1],psin1[2],psin1[3])
    f2 = fun(i,psin2[0],psin2[1],psin2[2],psin2[3])
    y = cauchy2d(t,f1,f2,fs(i)/1,0,fa(i)/1)
    tm1.append(y)



time = np.column_stack(tm1) # Final simulated sinogram
rt= 0.01


# Reconstruct images using Inverse Radon Transformation

reconstruction1 = iradon(time,theta=thetas,circle=True) # - simulated sinogram (time)
reconstruction2 = iradon(data0,theta=theta,circle=True) # - original sinogram (data0)
reconstruction3 = iradon(sim,theta=theta,circle=True) # - curve-fitted sinogram (sim)

# Display raw sinograms
fig = plt.figure(figsize=(30, 10))
ax1 = plt.subplot2grid((2, 4),(1,0),colspan=4, rowspan=1)
ax2 = plt.subplot2grid((2, 4),(0,0),colspan=4, rowspan=1)

ax1.imshow(time,aspect='auto',cmap='gray')
ax2.imshow(sim,aspect='auto',cmap='gray')
plt.show()

# Display reconstructed images

fig2 = plt.figure(figsize=(30, 7))
ax2 = plt.subplot2grid((2, 6),(0,0),colspan=2, rowspan=2)
ax3 = plt.subplot2grid((2, 6),(0,2),colspan=2, rowspan=2)
ax1 = plt.subplot2grid((2, 6),(0,4),colspan=2, rowspan=2)

ax1.imshow(reconstruction1,aspect='auto',cmap="gray")
ax1.set_title('Simulated Position')
ax1.set_xticks([])
ax1.set_yticks([])
ax2.imshow(reconstruction2,aspect='auto',cmap="gray")
ax2.set_xticks([])
ax2.set_yticks([])
ax2.set_title('Original Image')
ax3.imshow(reconstruction3,aspect='auto',cmap="gray")
ax3.set_xticks([])
ax3.set_yticks([])
ax3.set_title('Curve Fit Enhacement')
plt.savefig('figures/enhance.png')
plt.show()
