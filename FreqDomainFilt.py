import numpy
import matplotlib.pyplot as mplplt
import matplotlib.image as mplimg

##Upload Images to Test
#img = mplimg.imread(r'testImage.jpg')
img = mplimg.imread(r'Monarch.jpg')

##Convert Image Into Grayscale
def rgb2gray(image):
    imgNew = numpy.dot(image[...,:3], [0.2126, 0.7152, 0.0722])
    return imgNew

grayImg = numpy.round(rgb2gray(img))

#Display Grayscaled Iamge
mplplt.imshow(grayImg, cmap = 'gray')
mplplt.title("Original Image in Grayscale")
mplplt.show()

a = [[1,2,3],[4,5,6],[7,8,9]]
print("a: ", a)
print("aMax: ", numpy.amax(a))
print("a Normalized:", a/numpy.amax(a)*255)
print(a[1][0])

#Find the Size of Each Dimension of the Original Image
imgXSize, imgYSize = numpy.shape(grayImg)

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

def DFT(centeredimg):
    DFTImage = numpy.fft.fft2(centeredImg)
    return DFTImage

DFTImg = DFT(centeredImg)
magDFTImg = numpy.abs(DFTImg)
#print(numpy.amax(magDFTImg))
#magDFTImg = magDFTImg/numpy.amax(magDFTImg)*255
magDFTImg = 20*numpy.log(magDFTImg)
print("magDFTImgmax:", numpy.amax(magDFTImg))
magDFTImg = magDFTImg.astype(int)
print("magDFTImg: " , magDFTImg)

mplplt.imshow(magDFTImg, cmap = 'gray')
mplplt.title("DFT of Image")
mplplt.show()



#magDFTIMG =
#normalizedImg = a/(numpy.amax(a))
#print(normalizedImg)

