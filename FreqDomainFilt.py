import numpy
import matplotlib.pyplot as mplplt
import matplotlib.image as mplimg

##Upload Images to Test
img = mplimg.imread(r'testImage.jpg')
#img = mplimg.imread(r'Monarch.jpg')

##Convert Image Into Grayscale
def rgb2gray(image):
    imgNew = numpy.dot(image[...,:3], [0.2126, 0.7152, 0.0722])
    return imgNew

grayImg = numpy.round(rgb2gray(img))

#Display Grayscaled Image
mplplt.imshow(grayImg, cmap = 'gray')
mplplt.title("Original Image in Grayscale")
mplplt.show()

a = [[1,0,0,1,2,3,4,5,6,7,8,9,0,0,0],[2,0,0,1,2,3,4,5,6,7,8,9,0,0,0], [3,0,0,1,2,3,4,5,6,7,8,9,0,0,0]]
print("Dimensions: ",numpy.shape(a))
imgXSize = 15
imgYSize = 3
#print(aNew)



#print(numpy.delete(a, range(0,2), 1))
#print("a: ", a)
#print("aMax: ", numpy.amax(a))
#print("a Normalized:", a/numpy.amax(a)*255)
#print(a[1][0])

#Find the Size of Each Dimension of the Original Image
imgYSize, imgXSize = numpy.shape(grayImg)

def padImage(img, imgXSize, imgYSize):
    padOneSideX = numpy.int(imgXSize/2) #numpy.int finds integer floor of argument
    padOneSideY = numpy.int(imgYSize/2)
    paddedImg = numpy.lib.pad(img, ((padOneSideY,padOneSideY),(padOneSideX,padOneSideX)), 'constant', constant_values=0) #paddedImg = numpy.lib.pad(a, ((5,5), (2,2)), 'constant', constant_values=0)  # numpy.int finds integer floor of argument
    return paddedImg

paddedImg = padImage(grayImg, imgXSize, imgYSize)
print(paddedImg)

mplplt.imshow(paddedImg, cmap = 'gray')
mplplt.title("Padded Image")
mplplt.show()

def centerImage(img, imgXSize, imgYSize):
    centeredImg = numpy.zeros((imgXSize,imgYSize))
    for x in range(0, imgXSize):
        for y in range(0, imgYSize):
            centeredImg[x,y] = img[x,y] * numpy.power(-1, x+y)
    return centeredImg

paddedXSize, paddedYSize = numpy.shape(paddedImg)

centeredImg = centerImage(paddedImg,paddedXSize,paddedYSize)
print(centeredImg)
print(centeredImg[1,2])

#Compute the DFT
def DFT(centeredimg):
    DFTImage = numpy.fft.fft2(centeredImg)
    return DFTImage

DFTImg = DFT(centeredImg)
magDFTImg = numpy.abs(DFTImg)
magDFTImg = 20*numpy.log(magDFTImg)
magDFTImg = magDFTImg.astype(int)

mplplt.imshow(magDFTImg, cmap = 'gray')
mplplt.title("DFT of Image")
mplplt.show()

DFTYSize, DFTXSize = numpy.shape(paddedImg)
#Function for Creating any MxN Size Gaussian Kernal based on Gaussian Formula
def createGaussian(XSize, YSize, sigma): #Accepts Side Lengths Only
    gaussian = numpy.zeros((YSize, XSize))
    sideXMax = numpy.int(numpy.floor(XSize / 2))
    sideXMin = -sideXMax
    sideYMax = numpy.int(numpy.floor(YSize / 2))
    sideYMin = -sideYMax
    if sideXMax % 2==0:
        sideXMax -= 1
    if sideYMax % 2 == 0:
        sideYMax -= 1
    total = 0
    for s in range(sideYMin, sideYMax+1):
        for t in range(sideXMin, sideXMax + 1):
            gaussian[s+sideYMax][t+sideXMax] = numpy.exp(-(s*s+t*t)/(2*sigma*sigma))
            total += gaussian[s+sideYMax][t+sideXMax]
    return gaussian

gauss = createGaussian(DFTXSize,DFTYSize,100)

print("SizeDFT:", numpy.shape(DFTImg))
print("SizeGauss:", numpy.shape(gauss))
gaussianFiltDFT = gauss*DFTImg

magDFTImg = numpy.abs(gaussianFiltDFT)
magDFTImg = 20*numpy.log(magDFTImg)
print("magDFTMax: ",numpy.amax(magDFTImg))
magDFTImg = magDFTImg.astype(int)

'''
mplplt.imshow(magDFTImg, cmap = 'gray')
mplplt.title("Gaussian'd DFT of Image")
mplplt.show()
'''

#Compute the iDFT
def iDFT(fft):
    filteredImg = numpy.fft.ifft2(fft)
    return filteredImg

filteredImg = iDFT(gaussianFiltDFT)
filteredImg = numpy.abs(filteredImg)
filteredImg = filteredImg.astype(int)
print(filteredImg)

mplplt.imshow(filteredImg, cmap = 'gray')
mplplt.title("Filtered Image")
mplplt.show()

def unpad(img, imgXSize, imgYSize):
    padOneSideX = numpy.int(imgXSize/2) #numpy.int finds integer floor of argument
    padOneSideY = numpy.int(imgYSize/2)
    imgUnpad = numpy.delete(img, range((imgXSize*2 - padOneSideX), imgXSize*2), 1)
    imgUnpad = numpy.delete(imgUnpad, range((imgYSize*2 - padOneSideY), imgYSize*2), 0)
    imgUnpad = numpy.delete(imgUnpad, range(0, padOneSideY), 0)
    imgUnpad = numpy.delete(imgUnpad, range(0, padOneSideX), 1)
    return imgUnpad

unpaddedFilteredImg = unpad(filteredImg, imgXSize,imgYSize)
mplplt.imshow(unpaddedFilteredImg, cmap = 'gray')
mplplt.title("Unpadded Filtered Image")
mplplt.show()



print("Hello World")
