from skimage.transform import iradon_sart
import numpy as np
import matplotlib.pyplot as plt

def loadfile(fname,col):
    return np.loadtxt(fname, usecols=(col,))
data1 = []
t = 361

for i in range(0,t,18):
    x= loadfile("Scan4_"+str(i)+"Deg.txt",1)
    print(x)
    data1.append(x)

data= np.column_stack(data1)
theta = np.arange(0,t,18)

y= loadfile("Scan4_0Deg.txt",0) + 10
reconstruction = iradon_sart(data, theta=theta,relaxation=0.15)
errodata = data**(1/2)
p = [0,1,2,3,4,5,6,7]
color = ['b', 'g', 'r', 'c', 'm', 'y', 'k','silver','navy','chocolate','lime']
j=0


fig = plt.figure(figsize=(25, 10))
ax1 = plt.subplot2grid((2, 4),(0,2),colspan=2)
ax2 = plt.subplot2grid((2, 4),(0,0),colspan=2, rowspan=2)
ax3 = plt.subplot2grid((2, 4),(1,2),colspan=2)

print(data)
ax1.imshow(data, cmap='gray',extent=[0,t-1,-10,10],aspect='auto',interpolation='quadric')
ax1.set_ylabel('Projection Axis (mm)')
ax1.set_title("Projection Sinogram")
ax1.set_xlabel("Projection Angles (deg)")
ax2.imshow(reconstruction, cmap='gray',aspect='auto')
ax2.set_xticks([])
ax2.set_yticks([])
ax2.set_title("Reconstruction",fontsize=15)
for i in p:
    ax3.errorbar(y,data[:, i],color = color[i],yerr=errodata[:, i])
ax3.legend(theta[(p)],ncol=2)
ax3.set_xlabel('Projection Axis (mm)')
ax3.set_ylabel('Intensity (counts)')
ax3.set_title("Projections")
fig.subplots_adjust(wspace=0.3, hspace=0.4)
plt.show()