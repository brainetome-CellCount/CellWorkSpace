#%%
import numpy as np
import os
from glob import *

from PIL import Image
import matplotlib.pyplot as plt
import scipy

from xml.dom.minidom import parse
import xml.dom.minidom
import xml
import math

from skimage.transform import resize
import pandas as pd
from PIL import ImageDraw
import operator
from tqdm import *

#%%
# 读取xml文件 生成label list
def label_person(PersonFile):
    label_p = []

    x = []
    DOMTree = xml.dom.minidom.parse(PersonFile)
    collection = DOMTree.documentElement
    objs = collection.getElementsByTagName("Contour")
    for obj in objs:
        x.append(obj.getElementsByTagName("Pt")[0].childNodes[0].data)
    tmplst = [[m.split(',')[0],m.split(',')[1]] for m in x]
    label_p.append(tmplst)

    # turn str to int
    label_p = [[int(l[0].split('.')[0]), int(l[1].split('.')[0])] for l in label_p[0]]
    
    return label_p

# 去除重复标记
def deletrd(label_lst):
    tmp = label_lst.copy()
    for xy in label_lst:
        if xy not in tmp: continue
        x = xy[0]
        y = xy[1]
        for xy2 in label_lst:
            if operator.eq(xy, xy2): continue
            if xy2 not in tmp: continue
            if (x-xy2[0])**2+(y-xy2[1])**2 < 80:
                tmp.remove(xy2)
    
    return tmp

# 匹配坐标
def MatchCord(laborlst, gturelst):
    matchNum = 0
    for lb in laborlst:
        xlb = lb[0]
        ylb = lb[1]
        for gtl in gturelst:
            xgtl = gtl[0]
            ygtl = gtl[1]
            if (xlb-xgtl)**2+(ylb-ygtl)**2 < 100:
                matchNum = matchNum + 1
                break
    return matchNum

#%%
yzy_pth = 'F:/三个月标记数据/第二个月/yzy_标记_第一个月/001cell'

label_yzy = label_person(yzy_pth+'/VOIPoint_0.xml')
label_yzy = deletrd(label_yzy)

print(label_yzy)
#%%
pth = 'F:/三个月标记数据/第三个月/wsf_标记_第三个月'
savepth = 'F:/三个月标记数据/label/第三个月/wsf/'

labelpth = []
for f in os.walk(pth):
    if 'wsf' in f[0]:
        labelpth.append(f[0])

for lpth in labelpth[1:]:
    # 读取xml文件得到label
    try:
        label_data = label_person(lpth+'/VOIPoint_0.xml')
        label_data = deletrd(label_data)
    except:
        continue
    # 保存label为csv文件
    df = pd.DataFrame(columns=['x', 'y'])
    df['x'] = np.array(label_data)[:,0]
    df['y'] = np.array(label_data)[:,1]
    # save
    os.makedirs(savepth, exist_ok=True)
    df.to_csv(savepth+lpth.split('\\')[-1]+'.csv', index=False)
#%%
# 最大亮度投影
rootpth = 'F:/三个月标记数据/lxy_原始图像/第三个月'
sigimgpth = 'F:/三个月标记数据/raw_resize_img/lxy_rawimg/'
# sigimgpth = 'F:/三个月标记数据/raw_img/lxy_rawimg_slice/'
os.makedirs(sigimgpth, exist_ok=True)

# get all img file names
imgroot = list()
for i, m in enumerate(sorted(os.walk(rootpth))):
    if i > 0: break
    imgroot = m[1]
print(len(imgroot), imgroot[0])

cnt = 0
for imgfile in imgroot:
    tiflst = glob(rootpth+'/'+imgfile+'/*.tif')
    tiflst = [f for f in tiflst]
    tiflst = np.array(tiflst)[[0,2,4]].tolist()
    for i, imglst in enumerate(tiflst):
        tifimg = Image.open(imglst)
        tifimgarr = np.array(tifimg).reshape((1300,1300,1))
        if i == 0:
            tifall = tifimgarr
        else:
            tifall = np.concatenate((tifall, tifimgarr), axis=-1)
    tifall = tifall.max(axis=-1).reshape((1300,1300))
    tifall = resize(tifall, (256,256), mode='reflect')
    scipy.misc.toimage(tifall, high=tifall.max(), mode='F').save(sigimgpth+'/'+imgfile+'.tif')
    # scipy.misc.toimage(tifall, high=tifall.max(), mode='F').save(sigimgpth+'/'+str(1000+cnt+1)[1:]+'cell.tif')
    cnt += 1
#%%
# 定义高斯核函数
# defined Gaussion filter
def Gussion(m, n, sigma):
    sigma = sigma**2
    return math.exp(-(m**2+n**2)/(2*sigma))/(2*math.pi*sigma)

def Gaussion_filter(kernel_size0=3, kernel_size1=3, sigma = 4):
    H= []
    bs0 = (kernel_size0-1)/2
    bs1 = (kernel_size1-1)/2
    m = np.linspace(-bs0,bs0,kernel_size0)
    n = np.linspace(-bs1,bs1,kernel_size1)
    for x in m:
        h = [Gussion(x,f,sigma) for f in n]
        H.append(h)
    return H/np.sum(H)
#%%
# 密度图生成函数
# density map genrate
def density_map(imgpath, ksize, imgsize, imgorcsv=0):
    hksize = ksize // 2
    if not imgorcsv:
        dotimg = Image.open(imgpath)
        imgarr = np.array(dotimg, dtype='float')
        imgarr = np.pad(np.array(imgarr).T[0,:,:], ((hksize,hksize),(hksize,hksize)), 'constant', constant_values=(0,0))
        dotlist = np.argwhere(imgarr > 0).tolist()
    else:
        csvf = pd.read_csv(imgpath)
        dotlist = zip(csvf['x'].astype('int'), csvf['y'].astype('int'))
        
    h, w = imgsize+ksize-1, imgsize+ksize-1
    h = int(h)
    w = int(w)
    f_size = ksize
    
    imgarr = np.zeros((h,w))
    for x, y in dotlist:
        # 当使用csv读取坐标时执行
        if imgorcsv:
            x = np.round(x*255/1300)
            y = np.round(y*255/1300)
            x = x+2
            y = y+2
        
        x1 = int(x - hksize)
        y1 = int(y - hksize)
        x2 = int(x + hksize)
        y2 = int(y + hksize)
        
        #print(x1,y1,x2,y2,point[0],point[1],f_sizeW,f_sizeH)
        imgarr[y1:y2+1, x1:x2+1] += Gaussion_filter(ksize,ksize,1)
        
    return imgarr

#%%
# 读取坐标csv文件时执行的密度图生成操作
basepth = 'F:/three_lable/label/three_month/lxy'
spth = 'F:/三个月标记数据/dens_resize_img/第三个月/lxy/'
os.makedirs(spth, exist_ok=True)

imglist = os.listdir(basepth)
print(len(imglist),imglist)

for dens in imglist:
    # read image
    densimg = density_map(basepth+'/'+dens, 5, 256, imgorcsv=1)
    img = Image.fromarray(densimg).crop((2,2,260-2,260-2))
    img = np.array(img)
    scipy.misc.toimage(img, high=img.max(), mode='F').save(spth+dens.split('_')[0]+'.tiff')

#%%
# img标记数量
denspth = 'F:/three_lable/label/one_month/lxy_lyy/'
dspth = 'D:/cell_code/data/cells_Data/allcelldata/train/label/'

denslist = os.listdir(denspth)

cellnum = []
cellnumber = []
for fpth in denslist:
    tmpdf = pd.read_csv(denspth+fpth)
    cellnum.append(tmpdf.shape[0])
    cellnumber.append(fpth.split('_')[0])

densdf = pd.DataFrame(columns=['id', 'num'])
densdf['id'] = cellnumber
densdf['num'] = cellnum
densdf.to_csv(dspth+'label.csv', index=False)
#%%
from sklearn.model_selection import train_test_split

a = np.array([1,2,3,4,5,6])
y = np.array([1,0,1,1,0,0])
trx, tex, trys, teys = train_test_split(a, y, test_size=0., random_state=29)

print(trx, trys)

#%%
## 获取mask细胞图像
imgpth = 'D:/cell_code/data/cells_Data_20/allcelldata/cell'
labpth = 'F:/z-stack/singleimg/newrealdens/order'
spth = 'F:/三个月标记数据/mask_imgs/gt/'
os.makedirs(spth, exist_ok=True)

flist = os.listdir(imgpth)
lablist = os.listdir(labpth)
for f in zip(flist, lablist):
    print(f[0], f[1])
    imgs = Image.open(imgpth+'/'+f[0])
    lab = pd.read_csv(labpth+'/'+f[1])
    mask = np.array([[0]*256 for _ in range(256)])  # 256x256
    for x, y in zip(lab.x.values, lab.y.values):
        '''
        x = np.round(x*255/1300)
        y = np.round(y*255/1300)
        x = x+2
        y = y+2
        x1 = int(x - 6)
        y1 = int(y - 6)
        x2 = int(x + 6)
        y2 = int(y + 6)
        '''
        tmp = np.full_like(mask[y1:y2+1, x1:x2+1], 1)
        mask[y1:y2+1, x1:x2+1] = tmp
    maskimg = np.multiply(imgs, mask)
    scipy.misc.toimage(maskimg, high=maskimg.max(), mode='F').save(spth+f[0])
#%%
## 获取mask细胞图像
imgpth = 'F:/z-stack/singleimg/raw_img/test20/'
labpth = 'F:/z-stack/singleimg/raw_img/label_state/gt/'
spth = 'F:/z-stack/singleimg/raw_img/mask/gt/'
os.makedirs(spth, exist_ok=True)

imglab = os.listdir(labpth+'img_lab_dict/')

cnt = 0
for imlb in tqdm(imglab):
    imgid = imlb.split('_')[0]
    cnt += 1
    idxy = np.load(labpth+'idxy_dict.npy').item()
    imglab = np.load(labpth+'img_lab_dict/'+imlb).item()
    imgname = list(imglab.keys())

    imgs = Image.open(imgpth+imgid+'.tif')
    mask = np.array([[0]*1300 for _ in range(1300)])  # 256x256
    for imgn in imgname:
        x, y = idxy[imgn][0], idxy[imgn][1]
        '''
        x = np.round(x*255/1300)
        y = np.round(y*255/1300)
        x = x+2
        y = y+2
        '''
        x1 = int(x - 36)
        y1 = int(y - 36)
        x2 = int(x + 36)
        y2 = int(y + 36)

        tmp = np.full_like(mask[y1:y2+1, x1:x2+1], 1)
        mask[y1:y2+1, x1:x2+1] = tmp
    maskimg = np.multiply(imgs, mask)
    scipy.misc.toimage(maskimg, high=maskimg.max(), mode='F').save(spth+imgid+'_mask.tif')

#%%
"""
    resize
"""
imgpth = 'F:/z-stack/singleimg/raw_img/mask/gt/'
spth = 'F:/z-stack/singleimg/raw_img/mask/gt_resize/'

imgn = os.listdir(imgpth)

for imn in imgn:
    tifimg = Image.open(imgpth+imn)
    tifimgarr = np.array(tifimg)

    tifall = resize(tifimgarr, (256,256), mode='reflect')
    scipy.misc.toimage(tifall, high=tifall.max(), mode='F').save(spth+imn)

#%%
