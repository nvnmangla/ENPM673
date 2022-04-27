import copy
import argparse
from copyreg import dispatch_table
from dis import dis
import glob
import cv2
from matplotlib import pyplot as plt
import numpy as np
import numpy.linalg as nl
import matplotlib.pyplot as plt
from tqdm import *




def normalize(pts):
    
    pts_mean = np.mean(pts, axis=0) 

    x_mean ,y_mean = pts_mean[0], pts_mean[1]

    x_cap = pts[:,0] - x_mean
    y_cap = pts[:,1] - y_mean
    
    #scaled so that the mean of distances from the origin to the points equals sqrt(2). 
    s = (2/np.mean(x_cap**2 + y_cap**2))**(1/2)

    T_scale = np.diag([s,s,1])

    T_trans = np.array([[1,0,-x_mean],[0,1,-y_mean],[0,0,1]])

    T = T_scale@T_trans

    pts_norm = (T@pts.T).T

    return  pts_norm, T
    


    
def RANSAC(feat):
   
    thresh = 0.02
    
    index = [i for i in range(feat.shape[0])] 
    n= 0  
    i = 0
    F = 0
    Sin = []
    for i in range(1000):
       
        S = []

        rand_p = np.random.choice(index, size=8)
       
        feat_ch = feat[rand_p,:,:]
        
        Ftemp = Fundamental(feat_ch)

        for j in range(feat.shape[0]):

            x1j = feat[j,:,1].reshape(1,3)
            x2j = feat[j,:,0].reshape(1,3)
            xFx = x2j@Ftemp@x1j.T
            
            if abs(xFx)< thresh:
                S.append(j)
        if len(S)>n:
            n = len(S)
            Sin = S.copy()
            print("n=",n)
            F = Ftemp
            
    
    ranfeat = feat[Sin,:,:]

    return F,ranfeat

def drawLines(img, lines):
    img2 =copy.deepcopy(img)
    _, c, _ = img.shape
    for r in lines:
        x0, y0 = map(int, [0, -r[2]/r[1]])
        x1, y1 = map(int, [c, -(r[2]+r[0]*c)/r[1]])
        cv2.line(img2, (x0, y0), (x1, y1), (0,255,255),2)
    return img2

def R_rect(E):
    U,S,V = nl.svd(E)
    W = np.array([[0,-1,0],[1,0,0],[0,0,1]])
    

    R = U@W@V
    
    return R


def Essential(F,K1,K2):
    E = (K2.T)@F@K1
    
    U,S,V = nl.svd(E)
    S = np.diag(S)
    
    S[2,2] = 0
    E = U@S@V
    return E



def Camera(E,K1,K2):
    U,S,V = nl.svd(E)
    W = np.array([[0,-1,0],[1,0,0],[0,0,1]])


    R1 = U@W@V
    R2 = U@(W.T)@V
    
    t = U[:,2]

    return (R1,t),(R1,-t),(R2,t),(R2,-t)

def Fundamental(feats):

    if feats.shape[0]> 7:
    
        x1_norm, T1 = normalize(feats[:,:,0])
        x2_norm, T2 = normalize(feats[:,:,1])
        
    else:
        print("not enough points")
        return 0

    A = np.zeros((feats.shape[0],9))
    for p in range(feats.shape[0]):
        x1,y1 = x1_norm[p,0],x1_norm[p,1]
        x1p,y1p = x2_norm[p,0],x2_norm[p,1]
        A[p,:] = [x1*x1p,x1*y1p,x1,y1*x1p,y1*y1p,y1,x1p,y1p,1]
    U,S,V = nl.svd(A,full_matrices=True)
   
    F = V.T[:,8]
    # F = F/F[8]
    F = F.reshape(3,3)

    U,S,V = np.linalg.svd(F)
    S = np.diag(S)
    S[2,2] = 0

    F = U@S@V
    F = F.reshape(3,3)
    
    
    F = (T2.T)@F@T1
    
    
    return F



##########################################################
path = "/home/naveen/ENPM673/project3/data/octagon/"
#########################################################

data = 2
if data == 1:
    K1=np.array([[1758.23,0,977.42],[0,1758.23,552.15],[0,0,1]])
    K2=np.array([[1758.23,0,977.42],[0,1758.23,552.15],[0,0,1]])

    doffs=0
    baseline=88.39
    width=1920
    height=1080
    ndisp=220
    vmin=55
    vmax=195

elif data ==2:
    K1=np.array([[1742.11,0,804.90], [0,1742.11,541.22], [0,0,1]])
    K2=np.array([[1742.11,0,804.90], [0,1742.11,541.22], [0,0,1]])
    doffs=0
    baseline=221.76
    width=1920
    height=1080
    ndisp=100
    vmin=29
    vmax=61

elif data ==3:
    K1=np.array([[1729.05,0,-364.24],[0,1729.05,552.22],[0,0,1]])
    K2=np.array([[1729.05,0,-364.24],[0,1729.05,552.22],[0,0,1]])
    doffs=0
    baseline=537.75
    width=1920
    height=1080
    ndisp=180
    vmin=25
    vmax=150




im1 = cv2.imread(path+'im0.png')
im2 = cv2.imread(path+'im1.png')

gray1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

disc = cv2.xfeatures2d.SIFT_create()

# Describing Features
kp1, des1 = disc.detectAndCompute(gray1, None)
kp2, des2 = disc.detectAndCompute(gray2, None)

bf = cv2.BFMatcher()
matches = bf.match(des1,des2)



# Describing Matches

# Sorting matches with least
matches = sorted(matches, key=lambda x: x.distance)[:100]

feat = np.ones((len(matches),3,2))


for i, match in enumerate(matches):
    feat[i,:2,0] = np.array(kp1[match.queryIdx].pt)
    feat[i,:2,1] = np.array(kp2[match.trainIdx].pt)

# normalize(im1,im2,src,dst)
im3 = cv2.drawMatches(im1, kp1, im2, kp2, matches, None,(0,0,255))


cv2.imwrite(path+"matchesbr.png", im3)


drawsrc1,drawdst1 =[],[]
 

for m in range(feat.shape[0]):
  
        drawsrc1.append(cv2.KeyPoint(feat[m,0,0],feat[m,1,0],1))
        drawdst1.append(cv2.KeyPoint(feat[m,0,1],feat[m,1,1],1))


F,new_feat = RANSAC(feat)
print(F)

drawsrc2,drawdst2 =[],[]

E = Essential(F,K1,K2)

p1,p2,p3,p4 = Camera(E,K1,K2)





imd1 = copy.deepcopy(im1)
imd2 = copy.deepcopy(im2)


# new_feat = new_feat.astype(np.int0)


for m in range(new_feat.shape[0]):
        
        drawsrc2.append(cv2.KeyPoint(new_feat[m,0,0],new_feat[m,1,0],1))
        drawdst2.append(cv2.KeyPoint(new_feat[m,0,1],new_feat[m,1,1],1))
        

matches = [cv2.DMatch(idx, idx, 1) for idx in range(len(drawsrc2))]
im3 = cv2.drawMatches(imd1,drawsrc2,imd2,drawdst2,matches,None,(255,0,0))


cv2.imwrite(path+"matchesar.png", im3)



i,j,_ = im1.shape

_ ,H1, H2 = cv2.stereoRectifyUncalibrated(new_feat[:,:2,0],new_feat[:,:2,1],F,(j,i))

 

F_rec = (nl.inv(H2).T)@F@nl.inv(H1)



dst11 = cv2.warpPerspective(im1,H1,(j,i))
dst22 = cv2.warpPerspective(im2,H2,(j,i))


epilinesl = cv2.computeCorrespondEpilines(new_feat[:,:2,0], 1, F_rec)
epilinesR = cv2.computeCorrespondEpilines(new_feat[:,:2,1], 2, F_rec)

epilinesl = epilinesl.reshape(-1, 3)
epilinesR = epilinesR.reshape(-1, 3)

lines1 = drawLines(dst11,epilinesl)
lines2 = drawLines(dst22,epilinesR)
for m in range(new_feat.shape[0]):
    cv2.circle(dst11,(int(new_feat[m,0,0]),int(new_feat[m,1,0])),4,(255,255,0),-1)
    cv2.circle(dst22,(int(new_feat[m,0,1]),int(new_feat[m,1,0])),4,(0,255,255),-1)

cv2.imwrite(path+"warp1.png", lines1)
cv2.imwrite(path+"warp2.png", lines2)



g1 = cv2.cvtColor(dst11, cv2.COLOR_BGR2GRAY)
g2 = cv2.cvtColor(dst22, cv2.COLOR_BGR2GRAY)

#resizing 

sim1 = cv2.resize(g1, (j,i), interpolation = cv2.INTER_AREA)
sim2 = cv2.resize(g2, (j,i), interpolation = cv2.INTER_AREA)

window = 11
a,b = sim1.shape

disparity = np.zeros_like(sim1)
depth = np.zeros_like(sim1)

for r in tqdm(range(a-window)):
    for c in range(b-window):
        min_ssd,min_x = 1e+10,0.5
        
        pixel = sim1[r:r+window,c:c+window]
        
        
        x1 = max(0,c-50)
        x2 = min(c+50,b-window)

        for k in range(x1,x2,window):
            p2 = sim2[r:r+window,k:k+window]
            d = np.square(pixel- p2)
            
            SSD = np.sum(d)

            if SSD < min_ssd:
                min_ssd = SSD
                min_x = k

        disp = np.abs(min_x-c)
        
        disparity[r,c]= disp



disparity = disparity[:a-window,:b-window]


# disparity = (disparity)* 255 / (np.max(disparity))

depth = K1[0,0]*baseline/(disparity+1e-10)


depth[depth > 100000] = 100000

# depth = (depth)* 255 / (np.max(depth))

plt.imshow(disparity,cmap="gray")

plt.savefig(path+"disparity.jpg")

plt.imshow(disparity, cmap='hot' ,interpolation='nearest')
plt.savefig(path+ ' disparitymap.jpg')

plt.imshow(depth,cmap="gray")
plt.savefig(path+"depth.png")

plt.imshow(depth, cmap='hot', interpolation='nearest')
plt.savefig(path+ ' depthMap.jpg')
plt.show()


# print(depth)

# # depth = (depth)/(np.max(depth)-np.min(depth))
# cv2.imwrite(path+"disparity.png",disparity)

# cv2.imwrite(path+"depth.png",depth)



         
