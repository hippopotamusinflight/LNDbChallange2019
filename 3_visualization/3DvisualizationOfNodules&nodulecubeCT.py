
# coding: utf-8

# In[ ]:


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from itertools import product, combinations
import pandas as pd 
import SimpleITK as sitk
import os


# In[ ]:


# extract the annotation file
pd.set_option('display.max_rows', None)
annotation = pd.read_csv("/global/home/hpc4535/LNDbChallenge/LNDb_original_data/LNDb_info/trainset_csv/trainNodules_gt.csv")
# extracting volumes of nodules 
V = annotation['Volume']
annotation.head()


# In[ ]:


## extract real nodules coordinates
RealNodul = pd.read_csv("/global/home/hpc4535/LNDbChallenge/CNNtest/realcube_w_coord.csv")
Xcoordi = RealNodul['x']
Ycoordi = RealNodul['y']
Zcoordi = RealNodul['z']
Dia = RealNodul['Diameter']
RealNodul


# In[ ]:


# Density Plot and Histogram of volumes of nodules
import seaborn as sns
sns.distplot(V, hist=True, kde=True, 
             bins=100, color = 'darkblue', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4})


# In[ ]:


## cuboid function for cube
def plot_cuboid(center, size):
    """
       Create a data array for cuboid plotting.


       ============= ================================================
       Argument      Description
       ============= ================================================
       center        center of the cuboid, triple
       size          size of the cuboid, triple, (x_length,y_width,z_height)
       :type size: tuple, numpy.array, list
       :param size: size of the cuboid, triple, (x_length,y_width,z_height)
       :type center: tuple, numpy.array, list
       :param center: center of the cuboid, triple, (x,y,z)
   """
    # suppose axis direction: x: to left; y: to inside; z: to upper
    # get the (left, outside, bottom) point
    import numpy as np
    ox, oy, oz = center
    l, w, h = size

    x = np.linspace(ox-l/2,ox+l/2,num=10)
    y = np.linspace(oy-w/2,oy+w/2,num=10)
    z = np.linspace(oz-h/2,oz+h/2,num=10)
    x1, z1 = np.meshgrid(x, z)
    y11 = np.ones_like(x1)*(oy-w/2)
    y12 = np.ones_like(x1)*(oy+w/2)
    x2, y2 = np.meshgrid(x, y)
    z21 = np.ones_like(x2)*(oz-h/2)
    z22 = np.ones_like(x2)*(oz+h/2)
    y3, z3 = np.meshgrid(y, z)
    x31 = np.ones_like(y3)*(ox-l/2)
    x32 = np.ones_like(y3)*(ox+l/2)

    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    ax = fig.gca(projection='3d')
    # outside surface
    ax.plot_wireframe(x1, y11, z1, color='b', rstride=1, cstride=1, alpha=0.2)
    # inside surface
    ax.plot_wireframe(x1, y12, z1, color='b', rstride=1, cstride=1, alpha=0.2)
    # bottom surface
    ax.plot_wireframe(x2, y2, z21, color='b', rstride=1, cstride=1, alpha=0.2)
    # upper surface
    ax.plot_wireframe(x2, y2, z22, color='b', rstride=1, cstride=1, alpha=0.2)
    # left surface
    ax.plot_wireframe(x31, y3, z3, color='b', rstride=1, cstride=1, alpha=0.2)
    # right surface
    ax.plot_wireframe(x32, y3, z3, color='b', rstride=1, cstride=1, alpha=0.2)


# In[ ]:


# creating the medical 3d images as cuboids 
def plot_cuboidImg(center, size):
    """
       Create a data array for cuboid plotting.


       ============= ================================================
       Argument      Description
       ============= ================================================
       center        center of the cuboid, triple
       size          size of the cuboid, triple, (x_length,y_width,z_height)
       :type size: tuple, numpy.array, list
       :param size: size of the cuboid, triple, (x_length,y_width,z_height)
       :type center: tuple, numpy.array, list
       :param center: center of the cuboid, triple, (x,y,z)
   """
    # suppose axis direction: x: to left; y: to inside; z: to upper
    # get the (left, outside, bottom) point
    import numpy as np
    ox, oy, oz = center
    l, w, h = size

    x = np.linspace(ox-l/2,ox+l/2,num=10)
    y = np.linspace(oy-w/2,oy+w/2,num=10)
    z = np.linspace(oz-h/2,oz+h/2,num=10)
    x1, z1 = np.meshgrid(x, z)
    y11 = np.ones_like(x1)*(oy-w/2)
    y12 = np.ones_like(x1)*(oy+w/2)
    x2, y2 = np.meshgrid(x, y)
    z21 = np.ones_like(x2)*(oz-h/2)
    z22 = np.ones_like(x2)*(oz+h/2)
    y3, z3 = np.meshgrid(y, z)
    x31 = np.ones_like(y3)*(ox-l/2)
    x32 = np.ones_like(y3)*(ox+l/2)

    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    ax = fig.gca(projection='3d')
    # outside surface
    ax.plot_wireframe(x1, y11, z1, color='black', rstride=1, cstride=1, alpha=0.2)
    # inside surface
    ax.plot_wireframe(x1, y12, z1, color='black', rstride=1, cstride=1, alpha=0.2)
    # bottom surface
    ax.plot_wireframe(x2, y2, z21, color='black', rstride=1, cstride=1, alpha=0.2)
    # upper surface
    ax.plot_wireframe(x2, y2, z22, color='black', rstride=1, cstride=1, alpha=0.2)
    # left surface
    ax.plot_wireframe(x31, y3, z3, color='black', rstride=1, cstride=1, alpha=0.2)
    # right surface
    ax.plot_wireframe(x32, y3, z3, color='black', rstride=1, cstride=1, alpha=0.2)


# In[ ]:


## Testing on one img
ImgPath = '/global/home/hpc4535/LNDbChallenge/LNDb_original_data/all_data'
filename = 'LNDb-0001.mhd'
imgtest = os.path.join(ImgPath,filename)
img = sitk.ReadImage(imgtest)
img_sz = img.GetSize()
img_spc = np.array(list(img.GetSpacing()))


# In[ ]:


print(img_spc)
print(img_sz)


# In[ ]:


#caculate img phsical length, width, height
physicalImgSize = img_spc * img_sz
axislen = physicalImgSize/2
print(axislen)


# In[ ]:


## taking first nodule as example to plot cube
i=0
center0 = [Xcoordi[i], Ycoordi[i], Zcoordi[i]]


# In[ ]:


## taking second nodule
i=1
center1 = [Xcoordi[i], Ycoordi[i], Zcoordi[i]]


# In[ ]:


print(center0)


# In[ ]:


# cube size set as 20x20x20
length = 20 
width = 20 
height = 20
# caculate img length, width, height and set up origin of img
lenImg = axislen[0]*2 
widImg = axislen[1]*2 
heiImg = axislen[2]*2
centerO = [0, 0, 0]

#plotting cubes in 3d medical image (cubes are based on nodules)
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_xlabel('X')
ax.set_xlim(-axislen[0], axislen[0])
ax.set_ylabel('Y')
ax.set_ylim(-axislen[1], axislen[1])
ax.set_zlabel('Z')
ax.set_zlim(-axislen[2], axislen[2])
# plotting cube based on the first nodule
plot_cuboid(center0, (length, width, height))
# plotting second nodule
plot_cuboid(center1, (length, width, height)) 
# plotting 3d image of origin 0x0x0
plot_cuboidImg(centerO,(lenImg, widImg, heiImg))
#ax.view_init(60, 35)
plt.title('Case_LNDb-0001_CubesOfNodulesZoomOut')
plt.show()


# In[ ]:


# ploting zoom in view of cube and nodules
fig = plt.figure()
ax = fig.gca(projection='3d')
#set up the axises
ax.set_xlabel('X')
ax.set_xlim(-70, 50)
ax.set_ylabel('Y')
ax.set_ylim(-150, -30)
ax.set_zlabel('Z')
ax.set_zlim(-60, 20)
# draw cube
#r = [-1, 1]
#for s, e in combinations(np.array(list(product(r, r, r))), 2):
#    if np.sum(np.abs(s-e)) == r[1]-r[0]:
#        ax.plot3D(*zip(s, e), color="b")


#radius of first nodule1 from diameter column of annotation file
r0 = Dia[0]/2
# draw sphere of first nodule
u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
x = r0*np.cos(u)*np.sin(v)+center0[0]
y = r0*np.sin(u)*np.sin(v)+center0[1]
z = r0*np.cos(v)+center0[2]
ax.plot_wireframe(x, y, z, color="r")

#radius of second nodule1
r1 = Dia[1]/2
# draw sphere of second nodule
u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
x = r1*np.cos(u)*np.sin(v)+center1[0]
y = r1*np.sin(u)*np.sin(v)+center1[1]
z = r1*np.cos(v)+center1[2]
ax.plot_wireframe(x, y, z, color="r")


# draw a point
#ax.scatter([center0[0]], [center0[1]], [center0[2]], color="g", s=100)

from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
plot_cuboid(center0, (length, width, height))
plot_cuboid(center1, (length, width, height))


#a = Arrow3D([0, 1], [0, 1], [0, 1], mutation_scale=20,
#            lw=1, arrowstyle="-|>", color="k")
#ax.add_artist(a)
#ax.view_init(60, 35)
plt.title('Case_LNDb-0001_CubesOfNodulesZoomIn_20x20x20')
plt.show()


# In[ ]:


# myshow() function

def myshow(img, title=None, margin=0.05, dpi=80):
    nda = sitk.GetArrayFromImage(img)
    spacing = img.GetSpacing()

    if nda.ndim == 3:
        # fastest dim, either component or x
        c = nda.shape[-1]

        # the the number of components is 3 or 4 consider it an RGB image
        if c not in (3, 4):
            nda = nda[nda.shape[0] // 2, :, :]

    elif nda.ndim == 4:
        c = nda.shape[-1]

        if c not in (3, 4):
            raise RuntimeError("Unable to show 3D-vector Image")

        # take a z-slice
        nda = nda[nda.shape[0] // 2, :, :, :]

    xsize = nda.shape[1]
    ysize = nda.shape[0]

    # Make a figure big enough to accommodate an axis of xpixels by ypixels
    # as well as the ticklabels, etc...
    figsize = (1 + margin) * xsize / dpi, (1 + margin) * ysize / dpi

    plt.figure(figsize=figsize, dpi=dpi, tight_layout=True)
    ax = plt.gca()

    extent = (0, xsize * spacing[0], ysize * spacing[1], 0)

    t = ax.imshow(nda, extent=extent, interpolation=None)

    if nda.ndim == 2:
        t.set_cmap("gray")

    if(title):
        plt.title(title)

    plt.show()


# In[ ]:


## visualize the first case CT scan
myshow(img,title='CaseLNDb-0001_CTslide', margin=0.04, dpi=80 )


# In[ ]:


### visualize nodules of first case LNDb-0001 CT 
case0001_n1_20='/global/home/hpc4535/LNDbChallenge/LNDb_original_data/single_case/real_images_LNDb-0001_0_pos_(-44.20345052166667,-119.07324219166668,-37.5)_20x20x20.npy'
case0001_n2_20='/global/home/hpc4535/LNDbChallenge/LNDb_original_data/single_case/real_images_LNDb-0001_1_pos_(25.8525390625,-126.9697265625,-45.5)_20x20x20.npy'
case0001_n2_10='/global/home/hpc4535/LNDbChallenge/LNDb_original_data/single_case/real_images_LNDb-0001_1_pos_(25.8525390625,-126.9697265625,-45.5)_10x10x10.npy'
case0001_n2_6='/global/home/hpc4535/LNDbChallenge/LNDb_original_data/single_case/real_images_LNDb-0001_1_pos_(25.8525390625,-126.9697265625,-45.5)_6x6x6.npy'


# In[ ]:


case0001_n1_20 = np.load(case0001_n1_20)
case0001_n2_20 = np.load(case0001_n2_20)
case0001_n2_10 = np.load(case0001_n2_10)
case0001_n2_6 = np.load(case0001_n2_6)


# In[ ]:


print('case0001_n1_20.shape is',case0001_n1_20.shape)
print('case0001_n2_20.shape is',case0001_n2_20.shape)
print('case0001_n2_10.shape is',case0001_n2_10.shape)
print('case0001_n2_6.shape is',case0001_n2_6.shape)


# In[ ]:


img0001_n1_20 = sitk.GetImageFromArray(case0001_n1_20)
img0001_n2_20 = sitk.GetImageFromArray(case0001_n2_20)
img0001_n2_10 = sitk.GetImageFromArray(case0001_n2_10)
img0001_n2_6 = sitk.GetImageFromArray(case0001_n2_6)


# In[ ]:


### visualize first nodule cube of case LNDb-0001 CT 
myshow(img0001_n1_20,title='LNDb-0001Nodule1_20x20x20Cube_CT', margin=15, dpi=80 )


# In[ ]:


### visualize second nodule cube of case LNDb-0001 CT 
myshow(img0001_n2_20,title='LNDb-0001Nodule2_20x20x20Cube_CT', margin=15, dpi=80 )


# In[ ]:


### visualize second nodule cube of case LNDb-0001 CT 
myshow(img0001_n2_10,title='LNDb-0001Nodule2_10x10x10Cube_CT', margin=32, dpi=80 )


# In[ ]:


### visualize second nodule cube of case LNDb-0001 CT 
myshow(img0001_n2_6,title='LNDb-0001Nodule2_6x6x6Cube_CT', margin=52, dpi=80 )


# In[ ]:


### visualize background cubes of different cases CT scan
case0001_bc1_20='/global/home/hpc4535/LNDbChallenge/LNDb_original_data/single_case/bkgd_images_LNDb-0001_1_pos_(110,110,110)_20x20x20_truncNormd.npy'
case0001_bc1_20 = np.load(case0001_bc1_20)
print('case0001_bc1_20.shape is',case0001_bc1_20.shape)


# In[ ]:


img0001_bc1_20 = sitk.GetImageFromArray(case0001_bc1_20)


# In[ ]:


### visualize first nodule cube of case LNDb-0001 CT 
myshow(img0001_bc1_20,title='LNDb-0001Background_20x20x20Cube_CT', margin=16, dpi=80 )


# In[ ]:


### visualize background cubes of different case 0267 CT scan
case0267_fn1='/global/home/hpc4535/LNDbChallenge/LNDb_original_data/single_case/fake_images_LNDb-0264_340_pos_(-74.3740234375,-185.3134765625,34.0)_20x20x20_norm.npy'
case0267_fn1 = np.load(case0267_fn1)
case0267_fn1.shape


# In[ ]:


case0267_fn1 = sitk.GetImageFromArray(case0267_fn1)


# In[ ]:


myshow(case0267_fn1,title='CaseLNDb-0264Nodule1_CT', margin=16, dpi=80 )

