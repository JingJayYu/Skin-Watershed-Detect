import cv2
import numpy as np


def HMax(src, dst, h, kernel):
    '''
    HMax H-maxima transform.
    '''
    msk = np.zeros(src.shape, np.uint8)
    temp1 = np.zeros(src.shape, np.uint8)
    temp2 = np.zeros(src.shape, np.uint8)
    
    cv2.subtract(src, np.zeros(src.shape, np.uint8)+h, msk)#msk=src-h
    cv2.min(src, msk, dst)
    cv2.dilate(dst, kernel, dst)
    cv2.min(src, dst, dst)
    
    while True:
        temp1=np.copy(dst)
        cv2.dilate(dst, kernel, dst)
        cv2.min(src, dst, dst)
        #if temp1(i)==dst(i), than temp2(i)=0; else temp(i)=255
        cv2.compare(temp1, dst, cv2.CMP_NE, temp2)
        #for all i, if temp1(i)==dst(i), than break
        if cv2.sumElems(temp2)[0]==0:
            break
 
    return dst

def ExtendedHMax(src, dst, h, kernel):
    '''
    ExtendedHMax computes the extended-maxima transform, which
    is the regional maxima of the H-maxima transform.
    '''
    src_hmax_0 = np.zeros(src.shape, np.uint8)
    src_hmax_1 = np.zeros(src.shape, np.uint8)
    
    
    HMax(src, src_hmax_0,   h, kernel)
    HMax(src, src_hmax_1, h+150, kernel)
    
    cv2.subtract(src_hmax_0, src_hmax_1, dst)#dst=src_hmax_0-src_hmax_1
    
    return dst

def RegionalMax(src, dst, kernel):
    '''
    computes the regional maxima of src.
    '''
    ExtendedHMax(src, dst, 1, kernel)
    return dst

img = cv2.imread(r"C:\Users\JingJ\Desktop\skin_watershed\test.png")
imgsrc = img.copy()
img_gray = cv2.split(cv2.cvtColor(img,cv2.COLOR_BGR2LAB))[0] # get CS L

clahe = cv2.createCLAHE()
clahe_img = clahe.apply(img_gray)
gaussian_img = cv2.GaussianBlur(clahe_img, (5,5), 10)

kernel = np.ones((3, 3), np.uint8)
top_hat = cv2.morphologyEx(gaussian_img, cv2.MORPH_TOPHAT,kernel)
black_hat = cv2.morphologyEx(gaussian_img, cv2.MORPH_BLACKHAT, kernel)
morphology_trans = cv2.add(gaussian_img, top_hat)
morphology_trans = cv2.subtract(morphology_trans, black_hat)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
eMax = np.zeros(morphology_trans.shape, np.uint8)
ExtendedHMax(morphology_trans, eMax, 7, kernel)

_, morphology_threshold = cv2.threshold(morphology_trans, 127, 255 , cv2.THRESH_BINARY)
_, eMax_threshold = cv2.threshold(eMax, 127, 255, cv2.THRESH_BINARY)


# reconstruct = reconstruction(eMax_threshold,morphology_threshold,method='dilation')

mask = morphology_threshold
marker = eMax_threshold
element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
while True:
    marker_pre = marker 
    dilation = cv2.dilate(marker, kernel=element)
    marker = cv2.bitwise_and(dilation, mask)
    if(marker_pre == marker).all():
        break
reconstruct = marker




# noise removal
kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(reconstruct,cv2.MORPH_OPEN,kernel, iterations = 2)
# sure background area
sure_bg = cv2.dilate(opening,kernel,iterations=2)
# Finding sure foreground area
dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,0)
ret, sure_fg = cv2.threshold(dist_transform,0.1*dist_transform.max(),255,0)
# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg,sure_fg)

# Marker labelling
ret, markers = cv2.connectedComponents(sure_fg)
# Add one to all labels so that sure background is not 0, but 1
markers = markers+1
# Now, mark the region of unknown with zero
markers[unknown==255] = 0

markers = cv2.watershed(img,markers)
line = np.zeros((720,720), np.uint8)


for i in range(720):
    for j in range(720):
        if(markers[i][j] == -1):
            img[i][j][0] = 255
            img[i][j][1] = 0
            img[i][j][2] = 0
            line[i][j] = 255
            line[i][j] = 255
            line[i][j] = 255

cv2.imshow("img", img)
cv2.imshow("img src", imgsrc)
cv2.imshow("eMax_threshold", eMax_threshold)
cv2.imshow("img line", line)
cv2.imwrite("./result.png", img)
cv2.imwrite("./line.png", line)


cv2.waitKey(0)
cv2.destroyAllWindows()