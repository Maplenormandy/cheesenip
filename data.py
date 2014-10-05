import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2
import cv2

def scatterplot(data, data_name):
    """Makes a scatterplot matrix:
     Inputs:
         data - a list of data [dataX, dataY,dataZ,...];
                  all elements must have same length
         data_name - a list of descriptions of the data;
                  len(data) should be equal to len(data_name)
    Output:
         fig - matplotlib.figure.Figure Object
    """
    N = len(data)
    fig = plt.figure()
    for i in range(N):
        for j in range(N):
            ax = fig.add_subplot(N,N,i*N+j+1)
            if j == 0:
                ax.set_ylabel(data_name[i],size='12')
            if i == 0:
                ax.set_title(data_name[j],size='12')
            if i == j:
                ax.hist(data[i], 10)
            else:
                ax.scatter(data[j], data[i])
    return fig

class HandFinder:
    def __init__(self, data):
        # Post Processing - find eigenvalues + eigenvectors
        self.m1 = np.average(data, axis=1)
        cdata = np.zeros(np.shape(data))
        cdata[0] = data[0] - self.m1[0]
        cdata[1] = data[1] - self.m1[1]
        cdata[2] = data[2] - self.m1[2]
        self.m2m = np.zeros((3,3))
        for i in range(3):
            for j in range(3):
                self.m2m[i,j] = np.average(np.multiply(cdata[i], cdata[j]))

        self.m1a = np.array([[self.m1]*1280]*720)
        self.m2, self.m2e = np.linalg.eigh(self.m2m)
        self.m2v = np.dot(self.m2e, np.diag(np.sqrt(1.0 / self.m2)))

        print self.m1
        print self.m2m
        print self.m2
        print self.m2v

dataF = np.load('dataF.npy')
dataL = np.load('dataL.npy')
dataR = np.load('dataR.npy')


hfF = HandFinder(dataF)
hfL = HandFinder(dataL)
hfR = HandFinder(dataR)

"""
d1 = np.array([hfF.m1]*90)
cd = np.transpose(dataF) - d1
cd2 = np.einsum('ij,jk',cd,hfF.m2v)

d2 = np.transpose(dataF)
z2 = np.zeros(d2.shape)
for i in range(90):
    c = d2[i]
    c - hfF.m1
    d = np.dot(c,hfF.m2v)
    z2[i] = d

scatterplot(np.transpose(z2), ['X', 'Y', 'Z'])
plt.show()
"""

# Set up stuff
sampling = 16
size = 1280*720
sfArray = [chi2.sf(i/10.0, 3) for i in range(36*10)]

# Start the stuff
cap = [cv2.VideoCapture(i) for i in range(3)]

cap[0].set(3, 1280)
cap[0].set(4, 720)
cap[1].set(3, 640)
cap[1].set(4, 480)
cap[2].set(3, 640)
cap[2].set(4, 480)

avgF = None
avgL = None
avgR = None

#fgbg = [cv2.createBackgroundSubtractorMOG() for i in range(3)]

while True:
    rf = [cap[i].read() for i in range(3)]
    #fgmask = [fgbg[i].apply(rf[i][1]) for i in range(3)]
    frame = [rf[i][1] for i in range(3)]

    if avgF == None:
        avgF = np.zeros(frame[0].shape, dtype=frame[0].dtype)

    cv2.addWeighted(frame[0], 0.9, avgF, 0.1, 0, avgF)

    fd = cv2.absdiff(frame[0], avgF)
    fe = cv2.cvtColor(fd, cv2.COLOR_BGR2GRAY)

    f2 = np.tensordot(frame[0]-hfF.m1a, hfF.m2v, axes=[2,0])
    f3 = np.multiply(5.0-np.sum(np.square(f2),axis=2), fe) / 100.0



    cv2.imshow('frameF', f3)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

[cap[i].release() for i in range(3)]

cv2.destroyAllWindows()

