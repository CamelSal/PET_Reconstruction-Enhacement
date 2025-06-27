from scipy.signal import square
from skimage.transform import iradon
import numpy as np
import matplotlib.pyplot as plt


def loadfile(fname,col): #load the imformation of the given colum of the textfile as np a numpy array
    return np.loadtxt(fname, usecols=(col,))
data1 = [] #blank list to append each measurement for each column
t = 361
theta = np.arange(0,t,18)

for i in theta: #loads each file for the representing angel
    x= loadfile("Data/Scan4_"+str(i)+"Deg.txt",1)
    data1.append(x)

sinogram= np.column_stack(data1) # rearranges the data to form a projection sinogram obataned from the files


y= loadfile("Data/Scan4_0Deg.txt",0) + 10
reconstruction = iradon(sinogram, theta=theta) # reconstructed the image obatianed from the sinogram
# variable to generate the individual curves of coicidence signal
errodata = np.sqrt(data1) # error of each measurement which is proportional to the sqaureroot of the count
p = [0,1,2,3,4,5,6,7]
color = ['b', 'g', 'r', 'c', 'm', 'y', 'k','silver','navy','chocolate','lime']


# figure provides 3 different subplots
fig = plt.figure(figsize=(25, 10))
ax1 = plt.subplot2grid((2, 4),(0,2),colspan=2)
ax2 = plt.subplot2grid((2, 4),(0,0),colspan=2, rowspan=2)
ax3 = plt.subplot2grid((2, 4),(1,2),colspan=2)

#this plots the projection sinogram obtained from the multiple scans at different angels
ax1.imshow(sinogram, cmap='gray',extent=[0,t-1,-10,10],aspect='auto',interpolation='quadric')
ax1.set_ylabel('Projection Axis (mm)')
ax1.set_title("Projection Sinogram")
ax1.set_xlabel("Projection Angles (deg)")
# the recorstucted projection of the image done via an inversed radon trasformation
ax2.imshow(reconstruction, cmap='gray',aspect='auto')
ax2.set_xticks([])
ax2.set_yticks([])
ax2.set_title("Reconstruction",fontsize=15)
#plots of the projection of location vs count at specific angels whit their errors 

for i in p:
    ax3.errorbar(y,data1[i],color = color[i],yerr=errodata[i])
ax3.legend(theta[(p)],ncol=2)
ax3.set_xlabel('Projection Axis (mm)')
ax3.set_ylabel('Intensity (counts)')
ax3.set_title("Projections")
fig.subplots_adjust(wspace=0.3, hspace=0.4)
#plt.savefig('figures/original.png')
plt.show()
