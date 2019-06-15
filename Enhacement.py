import pylab as plb
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import factorial
from scipy import asarray as ar,exp
from skimage.transform import iradon_sart
from skimage.transform import iradon
from scipy.interpolate import interp1d

def loadfile(fname,col):
    return np.loadtxt(fname,comments=['D','#'] ,usecols=(col,))
data1 = []
t = 361


for i in range(0,t,18):
    #x1 = loadfile("Scan1_" + str(i) + "Deg.txt", 1)
    #x2 = loadfile("Scan2_" + str(i) + "Deg.txt", 1)
    #x3 = loadfile("Scan3_" + str(i) + "Deg.txt", 1)
    x4 = loadfile("Scan4_" + str(i) + "Deg.txt", 1)
    data1.append(x4)


data0= np.column_stack(data1)

theta = np.arange(0,t,18)


x0 = loadfile("Scan4_0Deg.txt",0)+ 10



def gaus(x,a,x0,sigma):
    return a*exp(-(x-x0)**2/(2*sigma**2))

def poiss(x,a,l):
    return a*(l**x)/factorial(x)*exp(-l)

def gaus2(x,a,x0,x01,sigma):
    return a*exp(-(x-x0)**2/(2*sigma**2))+a*exp(-(x-x01)**2/(2*sigma**2))

def cauchy(x,x0,s,y,a):
    return a/(np.pi*s*(1+((x-x0)/s)**2))+y

def cauchy2d(x,x02,x01,s,y,a):
    return a/(np.pi*s*(1+((x-x02)/s)**2))+y+a/(np.pi*s*(1+((x-x01)/s)**2))


def fun(x,a,p,y,c):
    return a*np.sin(x*p+y)+c

sim1=[]
prm = []
pk1= []
pk2= []
sd = []
a=[]
yoff=[]

t= np.arange(10.0,-10.0,-0.05)

popt=[0,0,1.739,254,11985]

for i in range(0,21):
    y= data0[:,i]
    popt,pcov = curve_fit(cauchy2d,x0,y,p0=popt,sigma=np.sqrt(y))
    f = cauchy2d(t,popt[0],popt[1],popt[2],popt[3],popt[4])
    #plt.plot(x0,y,'o')
    #plt.plot(t,f,'orange',linewidth=2)
    #plt.xlabel('Projection Axis (mm)')
    #plt.ylabel('Intensity (counts)')
    #plt.title("Cauchy Double Peak Curve Fit")
    #plt.legend(['Data Point','Cauchy Curve Fit'])
    #plt.show()
    sim1.append(f)
    pk1.append(popt[0])
    pk2.append(popt[1])
    sd.append(popt[2])
    yoff.append(popt[3])
    a.append(popt[4])
    prm.append(popt)

l1= [pk2[0]]
l2= [pk1[0]]
peak1=l1+pk1[1:11]+pk2[11:21]
peak2=l2+pk2[1:11]+pk1[11:21]

peak1=np.array(peak1)


sim =np.column_stack(sim1)

thetas= np.arange(0,360.1,0.1)

psin1,pcov = curve_fit(fun,theta,peak1,p0=[-4,0.02,-1,0])
psin2,pcov = curve_fit(fun,theta,peak2,p0=[4,0.016,-1.1,0])

csin1=fun(thetas,psin1[0],psin1[1],psin1[2],psin1[3])
csin2=fun(thetas,psin2[0],psin2[1],psin2[2],psin2[3])


s = 1.739
tm1 = []


fa = interp1d(theta,a,kind='cubic')
fs = interp1d(theta,sd,kind='cubic')

"""

plt.plot(theta,sd,'o')
plt.plot(thetas,fs(thetas))
plt.plot(theta,sd2,'o')
plt.plot(thetas,fs2(thetas))
plt.show()

plt.plot(theta,a,'o')
plt.plot(thetas,fa(thetas))
plt.plot(theta,a2,'o')
plt.plot(thetas,fa2(thetas))
plt.show()

plt.plot(theta,peak1,'o')
plt.plot(theta,peak2,'o')
plt.plot(thetas,csin1(thetas),'b')
plt.plot(thetas,csin2(thetas),'g')
plt.show()
"""


for i in thetas:
    f1 = fun(i,psin1[0],psin1[1],psin1[2],psin1[3])
    f2 = fun(i,psin2[0],psin2[1],psin2[2],psin2[3])
    y = cauchy2d(t,f1,f2,fs(i)/1,0,fa(i)/1)
    #plt.plot(y)
    #plt.show()
    tm1.append(y)



time = np.column_stack(tm1)
rt= 0.01

#reconstruction1 = iradon_sart(time,theta=thetas,relaxation=rt)
#reconstruction2 = iradon_sart(data0,theta=theta)
#reconstruction3 = iradon_sart(sim,theta=theta,relaxation=0.15)


reconstruction1 = iradon(time,theta=thetas,circle=True)
reconstruction2 = iradon(data0,theta=theta,circle=True)
reconstruction3 = iradon(sim,theta=theta,circle=True)

"""
fig = plt.figure(figsize=(30, 10))
ax1 = plt.subplot2grid((2, 4),(1,0),colspan=4, rowspan=1)
ax2 = plt.subplot2grid((2, 4),(0,0),colspan=4, rowspan=1)

ax1.imshow(time,aspect='auto',cmap='gray')
ax2.imshow(data0,aspect='auto',cmap='gray')
plt.show()
"""

fig2 = plt.figure(figsize=(30, 7))
ax2 = plt.subplot2grid((2, 6),(0,0),colspan=2, rowspan=2)
ax3 = plt.subplot2grid((2, 6),(0,2),colspan=2, rowspan=2)
ax1 = plt.subplot2grid((2, 6),(0,4),colspan=2, rowspan=2)

ax1.imshow(reconstruction1,aspect='auto',cmap="jet")
ax1.set_title('Simulated Position')
ax1.set_xticks([])
ax1.set_yticks([])
ax2.imshow(reconstruction2,aspect='auto',cmap="jet")
ax2.set_xticks([])
ax2.set_yticks([])
ax2.set_title('Original Image')
ax3.imshow(reconstruction3,aspect='auto',cmap="jet")
ax3.set_xticks([])
ax3.set_yticks([])
ax3.set_title('Curve Fit Enhacement')
plt.show()
