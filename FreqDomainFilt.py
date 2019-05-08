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

##Display Grayscaled Image
mplplt.imshow(grayImg, cmap = 'gray')
mplplt.title("Original Image in Grayscale")
mplplt.show()

#Find the Size of Each Dimension of the Original Image
imgYSize, imgXSize = numpy.shape(grayImg)

##Pad the Image such that the dimensions are twice of the original.
def padImage(img, imgXSize, imgYSize):
    padOneSideX = numpy.int(imgXSize/2) #numpy.int finds integer floor of argument
    padOneSideY = numpy.int(imgYSize/2)
    paddedImg = numpy.lib.pad(img, ((padOneSideY,padOneSideY),(padOneSideX,padOneSideX)), 'constant', constant_values=0) #paddedImg = numpy.lib.pad(a, ((5,5), (2,2)), 'constant', constant_values=0)  # numpy.int finds integer floor of argument
    return paddedImg

paddedImg = padImage(grayImg, imgXSize, imgYSize)

#Display Padded Image
mplplt.imshow(paddedImg, cmap = 'gray')
mplplt.title("Padded Image")
mplplt.show()

#Center the Padded Image to ensure that the Fourier Transform is Centered
def centerImage(img, imgXSize, imgYSize):
    centeredImg = numpy.zeros((imgXSize,imgYSize))
    for x in range(0, imgXSize):
        for y in range(0, imgYSize):
            centeredImg[x,y] = img[x,y] * numpy.power(-1, x+y)
    return centeredImg

paddedXSize, paddedYSize = numpy.shape(paddedImg)
centeredImg = centerImage(paddedImg,paddedXSize,paddedYSize)

#Compute the DFT of the centered and padded Image
DFTImg = numpy.fft.fft2(centeredImg)

#Function for Display the Magnitude Spectrum
def MagnitudeSpect(DFTImg):
    magDFTImg = numpy.abs(DFTImg)
    magDFTImg = 20*numpy.log(magDFTImg+0.0000001)
    magDFTImg = magDFTImg.astype(int)
    return magDFTImg

#Display the Magnitude Specturm of the Original Image
mplplt.imshow(MagnitudeSpect(DFTImg), cmap = 'gray')
mplplt.title("DFT of Original Image")
mplplt.show()

DFTYSize, DFTXSize = numpy.shape(paddedImg) #Finding the dimensions of the Fourier Transformed Image

#Function for Creating any MxN Size Gaussian Kernal based on Gaussian Formula
def createGaussian(XSize, YSize, sigma):
    gaussian = numpy.zeros((YSize, XSize))
    sideXMax = numpy.int(numpy.floor(XSize / 2)) #allows us to center the gaussian in the X dimension
    sideXMin = -sideXMax
    sideYMax = numpy.int(numpy.floor(YSize / 2)) #allows us to center the gaussian in the Y dimension
    sideYMin = -sideYMax
    if sideXMax % 2==0: #for handling an even XSize
        sideXMax -= 1
    if sideYMax % 2 == 0: #for handling an even YSize
        sideYMax -= 1
    total = 0
    for s in range(sideYMin, sideYMax+1):
        for t in range(sideXMin, sideXMax + 1):
            gaussian[s+sideYMax][t+sideXMax] = numpy.exp(-(s*s+t*t)/(2*sigma*sigma)) #Compute each gaussian per element
            total += gaussian[s+sideYMax][t+sideXMax]
    return gaussian

#Create the gaussian
gauss = createGaussian(DFTXSize,DFTYSize,100)

print("SizeDFT:", numpy.shape(DFTImg)) #Displays the dimensions of the DFT
print("SizeGauss:", numpy.shape(gauss)) #Displays the diemsnions of the Gaussian

#Filter the DFT using a gaussian
gaussianFiltDFT = gauss*DFTImg

#Display the Gaussian Filtered DFT
mplplt.imshow(MagnitudeSpect(gaussianFiltDFT), cmap = 'gray')
mplplt.title("Gaussian'd DFT of Image")
mplplt.show()

#Function for creating the laplacian
def Laplacian(XSize, YSize):
    laplacian = numpy.zeros((YSize, XSize))
    sideXMax = numpy.int(numpy.floor(XSize / 2))
    sideXMin = -sideXMax
    sideYMax = numpy.int(numpy.floor(YSize / 2))
    sideYMin = -sideYMax
    if sideXMax % 2 == 0:
        sideXMax -= 1
    if sideYMax % 2 == 0:
        sideYMax -= 1
    total = 0
    for s in range(sideYMin, sideYMax + 1):
        for t in range(sideXMin, sideXMax + 1):
            laplacian[s + sideYMax][t + sideXMax] = -(s * s + t * t)
            total += laplacian[s + sideYMax][t + sideXMax]
    return laplacian

#Compute the Laplacian
laplace = Laplacian(DFTXSize,DFTYSize)
print("SizeLaplace: ", numpy.shape(laplace)) #Displays the dimensions of the Laplacian


LoGFiltDFT = gaussianFiltDFT*laplace

mplplt.imshow(MagnitudeSpect(LoGFiltDFT), cmap = 'gray')
mplplt.title("LoG'd DFT of Image")
mplplt.show()

#Compute the iDFT of the Log Filtered Image
filteredImg = numpy.fft.ifft2(LoGFiltDFT)
filteredImg = numpy.abs(filteredImg)
filteredImg = filteredImg.astype(int)

#Display the Padded Filtered Image
mplplt.imshow(filteredImg, cmap = 'gray')
mplplt.title("Filtered Image (Padded)")
mplplt.show()

#Unpad an given image
def unpad(img, imgXSize, imgYSize):
    padOneSideX = numpy.int(imgXSize/2) #numpy.int finds integer floor of argument
    padOneSideY = numpy.int(imgYSize/2)
    imgUnpad = numpy.delete(img, range((imgXSize*2 - padOneSideX), imgXSize*2), 1)
    imgUnpad = numpy.delete(imgUnpad, range((imgYSize*2 - padOneSideY), imgYSize*2), 0)
    imgUnpad = numpy.delete(imgUnpad, range(0, padOneSideY), 0)
    imgUnpad = numpy.delete(imgUnpad, range(0, padOneSideX), 1)
    return imgUnpad

#Generate the unpadded Image
unpaddedFilteredImg = unpad(filteredImg, imgXSize,imgYSize)

#Display the Padded Image
mplplt.imshow(unpaddedFilteredImg, cmap = 'gray')
mplplt.title("Filtered Image (Unpadded)")
mplplt.show()

print("Hello World") #Code Runs to the End
