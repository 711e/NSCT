import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
from numpy.core.fromnumeric import reshape

from numpy.lib.type_check import imag
import weight_mask
from skimage import io
from scipy import ndimage
from scipy.signal import convolve2d
from scipy.signal import wiener
import math
import pywt
import utils
import os
import time

import scipy.io as scio

def myNSCTd(I,levels,pfiltername,dfiltername,type):

    [ph0, ph1, pg0, pg1] = atrousfilters(pfiltername)

    # filtersd = np.zeros((4))
    filtersd =  [[],[],[],[],
                [],[],[],[],
                [],[],[],[],
                [],[],[],[]]

    [dh1, dh2] = dfilters(dfiltername, 'd')
    dh1 = dh1 / np.sqrt(2)
    dh2 = dh2 / np.sqrt(2)

    filtersd[0] = modulate2(dh1, 'c', [])
    filtersd[1] = modulate2(dh2, 'c', [])

    [filtersd[2], filtersd[3]] = parafilters(dh1, dh2)

    clevels = len( levels )
    nIndex = clevels+1
    y = []
    for _ in range(nIndex):
        y.append([])
    Insp = []
    for _ in range(clevels):
        Insp.append([])
    for i in range(clevels):
        if type == 'NSCT':
            [Ilow, Ihigh] = NSPd(I, ph0, ph1, i)

        if levels[nIndex-2] > 0:
            Ihigh_dir = nsdfbdec(Ihigh, filtersd, levels[nIndex-2])
            y[nIndex-1] = Ihigh_dir
        else:
            y[nIndex-1] = xhigh
        nIndex = nIndex - 1
        I = Ilow
        Insp[i]=I
    y[0]=I
    Insct=y
    return [Insp,Insct]

def atrousfilters(fname):
    if fname == 'pyr':
        h0 = [
              [-0.003236043456039806, -0.012944173824159223, -0.019416260736238835],
              [-0.012944173824159223,  0.0625              ,  0.15088834764831843],
              [-0.019416260736238835,  0.15088834764831843 ,  0.3406092167691145]
             ]
        
        g1 = [[-0.003236043456039806, -0.012944173824159223, -0.019416260736238835],
            [-0.012944173824159223,-0.0625               ,-0.09911165235168155],
            [-0.019416260736238835, -0.09911165235168155  , 0.8406092167691145]]
        
        g0 = [ [-0.00016755163599004882, -0.001005309815940293, -0.002513274539850732, -0.003351032719800976],
            [-0.001005309815940293, -0.005246663087920392, -0.01193886400821893  ,  -0.015395021472477663],
            [-0.002513274539850732, -0.01193886400821893 ,  0.06769410071569153  ,   0.15423938036811946 ],
            [-0.003351032719800976, -0.015395021472477663 , 0.15423938036811946  ,   0.3325667382415921]]
        
        h1 = [  [0.00016755163599004882, 0.001005309815940293 ,  0.002513274539850732,  0.003351032719800976],
                [0.001005309815940293 , -0.0012254238241592198, -0.013949483640099517, -0.023437500000000007],
                [0.002513274539850732 , -0.013949483640099517 , -0.06769410071569153 , -0.10246268507148255],
                [0.003351032719800976 , -0.023437500000000007 , -0.10246268507148255 ,  0.8486516952966369]]
        h0 = np.array(h0)
        g0 = np.array(g0)
        h1 = np.array(h1)
        g1 = np.array(g1)
        g0 = np.hstack((g0, np.fliplr(g0[:,:-1])))
        g0 = np.vstack((g0, np.flipud(g0[:-1,:])))
        h0 = np.hstack((h0, np.fliplr(h0[:,:-1])))
        h0 = np.vstack((h0, np.flipud(h0[:-1,:])))

        g1 = np.hstack((g1, np.fliplr(g1[:,:-1])))
        g1 = np.vstack((g1, np.flipud(g1[:-1,:])))
        h1 = np.hstack((h1, np.fliplr(h1[:,:-1])))
        h1 = np.vstack((h1, np.flipud(h1[:-1,:])))
    
    return [h0,h1,g0,g1]

def dfilters(fname, type):
    if fname == 'pkva':
        beta = ldfilter(fname)
        [h0, h1] = ld2quin(beta)
        h0 = np.sqrt(2) * h0
        h1 = np.sqrt(2) * h1
        if type == 'r':
            f0 = modulate2(h1, 'b', [])
            f1 = modulate2(h0, 'b', [])
            h0 = f0
            h1 = f1

    return [h0, h1]

def ldfilter(fname):
    if fname == 'pkva':
        v = np.reshape([0.6300 ,  -0.1930 ,   0.0972 ,  -0.0526  ,  0.0272  , -0.0144], (1,6))
        v_ = np.fliplr(v)
        f = np.hstack((v_, v))
    return f

def ld2quin(beta):
    if beta.shape[0] != 1:
        print('The input must be an 1-D fitler')
    lf = beta.shape[1]
    n = int(lf / 2)
    sp = np.outer(beta, beta)
    h = qupz(sp, 1)
    h0 = np.copy(h)
    h0[2*n-1, 2*n-1] = h0[2*n-1, 2*n-1] + 1
    h0 = h0 / 2
    h1 = -1 * convolve2d(h,h0)
    h1[4*n-2, 4*n-2] = h1[4*n-2, 4*n-2] + 1
    return [h0, h1]

def qupz(x, type):
    if type == 1:
        x1 = resampz(x, 4, [])
        (m, n) = x1.shape
        x2 = np.zeros((2*m-1, n))
        j = 0
        for i in range(x2.shape[0]):
            if i % 2 == 0:
                x2[i, :] = x1[j]
                j += 1
        y = resampz(x2, 1, [])
    return y

def resampz(x, type, shift):
    if shift == []:
        shift = 1
    sx = x.shape
    if type == 3 or type == 4:
        y = np.zeros((sx[0], sx[1] + abs(shift * (sx[0] - 1))))
        if type != 3:
            a = np.arange(sx[0])
            shift2 = a * shift
        else:
            a = np.arange(sx[0])
            shift2 = a * (-shift)
        if shift2[-1] < 0:
            shift2 = shift2 - shift2[-1]
        for m in range(sx[0]):
            y[m, shift2[m]+np.arange(sx[1])] = x[m, :]
        start = 0
        u, s, v = np.linalg.svd(np.reshape(y[:, start], (1,-1)), full_matrices=False)
        while np.max(s) == 0:
            start = start + 1
            u, s, v = np.linalg.svd(np.reshape(y[:, start], (1,-1)), full_matrices=False)
        finish = y.shape[1]-1
        u, s, v = np.linalg.svd(np.reshape(y[:, finish], (1,-1)), full_matrices=False)
        while np.max(s) == 0:
            finish = finish - 1
            u, s, v = np.linalg.svd(np.reshape(y[:, finish], (1,-1)), full_matrices=False)
        y = y[:,start:finish+1]
    elif type == 1 or type == 2:
        y = np.zeros((sx[0] + abs(shift * (sx[1] - 1)), sx[1]))
        if type == 1:
            shift1 = np.arange(sx[1]) * (-shift)
        else:
            shift1 = np.arange(sx[1]) * (shift)
        if shift1[-1] < 0:
            shift1 = shift1 - shift1[-1]
        for n in range(sx[1]):
	        y[shift1[n]+np.arange(sx[0]), n] = x[:, n]
        start = 0
        u, s, v = np.linalg.svd(np.reshape(y[start, :], (1,-1)), full_matrices=False)
        while np.max(s) == 0:
            start = start + 1
            u, s, v = np.linalg.svd(np.reshape(y[start, :], (1,-1)), full_matrices=False)
        finish = y.shape[0]-1
        u, s, v = np.linalg.svd(np.reshape(y[finish, :], (1,-1)), full_matrices=False)
        while np.max(s) == 0:
            finish = finish - 1
            u, s, v = np.linalg.svd(np.reshape(y[finish, :], (1,-1)), full_matrices=False)
        y = y[start:finish+1]
    return y

def modulate2(x, type, center):
    if center == []:
        center = [0, 0]
    s = x.shape
    o = [int(s[0] / 2.)+1+center[0], int(s[1] / 2.)+1+center[1]]
    n1 = np.arange(1,s[0]+1) - o[0]
    n2 = np.arange(1,s[1]+1) - o[1]
    if type == 'c':
        m2 = [np.power(-1, abs(x)) for x in n2]
        m2 = np.array(m2)
        m2 = m2.reshape((1,-1))
        M = [s[0], 1]
        y = x * repmat(m2, M, [])
    elif type == 'r':
        m1 = [np.power(-1, abs(x)) for x in n1]
        m1 = np.array(m1)
        m1 = m1.reshape((-1,1))
        M = [1, s[1]]
        y = x * repmat(m1, M, [])
    elif type == 'b':
        m1 = [np.power(-1, abs(x)) for x in n1]
        m1 = np.array(m1)
        m1 = m1.reshape((-1,1))
        m2 = [np.power(-1, abs(x)) for x in n2]
        m2 = np.array(m2)
        m2 = m2.reshape((1,-1))
        m = np.outer(m1, m2)
        y = x * m

    return y

def repmat(A,M,N):
    if N == []:
        if len(M) > 1:
            siz = M
    if len(M) > 1 and len(siz) == 2:
        (m,n) = A.shape
        if m == 1 and siz[1] == 1:
            B = np.ones((siz[0],1))
            B = np.outer(B,A)
        elif n == 1 and siz[0] == 1:
            B = np.ones((1,siz[1]))
            B = np.outer(A,B)
    return B

def parafilters( f1, f2 ):
    y1 = [[], [], [], []]
    y2 = [[], [], [], []]

    y1[0] = modulate2(f1, 'r', [])
    y1[1] = modulate2(f1, 'c', [])

    y1[2] = np.array(y1[0]).T
    y1[3] = np.array(y1[1]).T

    y2[0] = modulate2(f2, 'r', [])
    y2[1] = modulate2(f2, 'c', [])
    y2[2] = np.array(y2[0]).T
    y2[3] = np.array(y2[1]).T
    for i in range(4):
        y1[i] = resampz( y1[i], i+1, [])
        y2[i] = resampz( y2[i], i+1, [])
    return [y1, y2]

def NSPd(I,h0,h1,level):
    index = []
    (m,n) = h0.shape
    Nh0= np.zeros((np.power(2,level) * m,np.power(2,level) * n))
    for i in range(0,Nh0.shape[0],np.power(2,level)):
        for j in range(0,Nh0.shape[1],np.power(2,level)):
            index.append([i,j])
    ind = 0
    for i in range(h0.shape[0]):
        for j in range(h0.shape[1]):
            Nh0[index[ind][0], index[ind][1]] = h0[i,j]
            ind += 1
    newh0 = Nh0[:(m-1)*np.power(2,level)+1,:(n-1)*np.power(2,level)+1]

    index = []
    (m,n) = h1.shape
    Nh1= np.zeros((np.power(2,level) * m,np.power(2,level) * n))
    for i in range(0,Nh1.shape[0],np.power(2,level)):
        for j in range(0,Nh1.shape[1],np.power(2,level)):
            index.append([i,j])
    ind = 0
    for i in range(h1.shape[0]):
        for j in range(h1.shape[1]):
            Nh1[index[ind][0], index[ind][1]] = h1[i,j]
            ind += 1
    newh1 = Nh1[:(m-1)*np.power(2,level)+1,:(n-1)*np.power(2,level)+1]

    # I = np.array(I, dtype=np.float32)
    # Ilow = imfilter(I, newh0, 'conv', 'symmetric', 'same')
    Ilow = cv2.filter2D(I, -1, newh0, borderType=cv2.BORDER_REFLECT)
    # Ihigh = imfilter(I, newh1, 'conv', 'symmetric', 'same')
    Ihigh = cv2.filter2D(I, -1, newh1, borderType=cv2.BORDER_REFLECT)
    return [Ilow,Ihigh]

def nsdfbdec( x, dfilter, clevels ):
    k1 = dfilter[0]
    k2 = dfilter[1]
    f1 = dfilter[2]
    f2 = dfilter[3]
    q1 = np.array([[1, -1],[1, 1]])
    y = [[],[],[],[]]
    if clevels == 1:
        [y[0], y[1]] = nssfbdec( x, k1, k2, [])
    else:
        [x1, x2] = nssfbdec( x, k1, k2, [])
        [y[0], y[1]] = nssfbdec( x1, k1, k2, q1 )
        [y[2], y[3]] = nssfbdec( x2, k1, k2, q1 )

        for l in range(3,clevels+1):
            y_old = y
            y = []
            for _ in range(np.power(2,l)):
                y.append([])
            for k in range(np.power(2,l-2)):
                slk = 2*int( (k) /2 ) - np.power(2,l-3) + 1
                mkl = 2*np.matmul(np.array([[np.power(2,l-3), 0],[ 0, 1 ]]),np.array([[1, 0],[-slk, 1]]))
                i = np.mod(k, 2)
                [y[2*k], y[2*k+1]] = nssfbdec( y_old[k], f1[i], f2[i], mkl )
            for k in range(np.power(2,l-2), np.power(2,l-1)):
                slk = 2 * int( ( k-np.power(2,l-2)-1 ) / 2 ) - np.power(2,l-3) + 1
                mkl = 2*np.matmul(np.array([[ 1, 0],[0, np.power(2,l-3) ]]),np.array([[1, -slk], [0, 1]]))
                i = np.mod(k, 2) + 2
                [y[2*k], y[2*k+1]] = nssfbdec( y_old[k], f1[i], f2[i], mkl )
    return y

def nssfbdec( x, f1, f2, mup ):
    if mup == []:
        # y1 = imfilter( x, f1,'symmetric' )
        y1 = cv2.filter2D(x, -1, f1, borderType=cv2.BORDER_REFLECT)
        # f1_ = scio.loadmat('./f1.mat')
        # f1_ = f1_['f1']
        # diff = np.abs(f1-f1_)
        # y2 = imfilter( x, f2,'symmetric' )
        y2 = cv2.filter2D(x, -1, f2, borderType=cv2.BORDER_REFLECT)
        return [y1, y2]
    if (mup == 1).all() or (mup == np.eye(2)).all():
        y1 = cv2.filter2D(x, -1, f1, borderType=cv2.BORDER_REFLECT)
        y2 = cv2.filter2D(x, -1, f2, borderType=cv2.BORDER_REFLECT)
        return [y1, y2]
    if mup.shape == (2,2):
        y1 = myzconv2( x, f1, mup )
        y2 = myzconv2( x, f2, mup )
    elif mup.shape == (1, 1):
        mup = mup * np.eye(2)
        y1 = myzconv2( x, f1, mup )
        y2 = myzconv2( x, f2, mup )
    return [y1, y2]

def myzconv2(Im,f,M):
    (fr,fc) = f.shape
    Nfstartr=min([1,1-(fr-1)*M[0,0],1-(fc-1)*M[1,0],1-(fr-1)*M[0,0]-(fc-1)*M[1,0]])
    Nfendr=max([1,1-(fr-1)*M[0,0],1-(fc-1)*M[1,0],1-(fr-1)*M[0,0]-(fc-1)*M[1,0]])
    Nfstartc=min([1,1-(fr-1)*M[0,1],1-(fc-1)*M[1,1],1-(fr-1)*M[0,1]-(fc-1)*M[1,1]])
    Nfendc=max([1,1-(fr-1)*M[0,1],1-(fc-1)*M[1,1],1-(fr-1)*M[0,1]-(fc-1)*M[1,1]])
    Nfr=Nfendr-Nfstartr+1
    Nfc=Nfendc-Nfstartc+1
    Nf=np.zeros((Nfr,Nfc))
    for i in range(fr):
        for j in range(fc):
            Nf[2-(i)*M[0,0]-(j)*M[1,0]-Nfstartr-1,2-(i)*M[0,1]-(j)*M[1,1]-Nfstartc-1]=f[i,j]
    # Imout = cv2.filter2D(Im, -1, Nf, borderType=cv2.BORDER_REFLECT)
    # Imout = cv2.filter2D(Im, -1, Nf, borderType=cv2.BORDER_WRAP)
    # Imout = cv2.filter2D(Im, -1, Nf, borderType=cv2.BORDER_REFLECT_101 )
    Imout = ndimage.convolve(Im, Nf, mode='wrap')
    return Imout

def myNSCTr(Insct,levels,pfiltername,dfiltername,type):
    [ph0, ph1, pg0, pg1] = atrousfilters(pfiltername)

    filtersr =  [[],[],[],[],
                [],[],[],[],
                [],[],[],[],
                [],[],[],[]]

    [dg1, dg2] = dfilters(dfiltername, 'r')

    dg1 = dg1 / np.sqrt(2)
    dg2 = dg2 / np.sqrt(2)

    filtersr[0] = modulate2(dg1, 'c', [])
    filtersr[1] = modulate2(dg2, 'c', [])
    [filtersr[2], filtersr[3]] = parafilters( dg1, dg2 )

    clevels = len( levels )
    nIndex = clevels+1

    Ilow=Insct[0]
    for i in range(clevels):
        if len(Insct[i+1]) > 1:
            Ihigh = nsdfbrec( Insct[i+1], filtersr )
        else:
            Ihigh = Insct[i+1]
        if type == 'NSCT':
            Ilow = NSPr(Ilow, Ihigh, pg0, pg1, clevels-i)
    Insctred=Ilow
    return Insctred

def nsdfbrec( x, dfilter ):
    len_x = 0
    for iterm in x:
        if iterm != []:
            len_x += 1
    clevels = int(np.log2( len_x ))

    k1 = dfilter[0]
    k2 = dfilter[1]
    f1 = dfilter[2]
    f2 = dfilter[3]

    q1 = np.array([[1, -1],[1, 1]])

    if clevels == 1:
        y = nssfbrec( x[0], x[1], k1, k2, [])
    else:
        for l in range(clevels,2,-1):
            for k in range(np.power(2,l-2)):
                slk = 2*int( (k) /2 ) - np.power(2,l-3) + 1
                mkl = 2*np.matmul(np.array([[np.power(2,l-3), 0],[ 0, 1 ]]),np.array([[1, 0],[-slk, 1]]))
                i = np.mod(k, 2)
                x[k] = nssfbrec( x[2*k], x[2*k+1], f1[i], f2[i], mkl )
            for k in range(np.power(2,l-2), np.power(2,l-1)):
                slk = 2 * int( ( k-np.power(2,l-2)-1 ) / 2 ) - np.power(2,l-3) + 1
                mkl = 2*np.matmul(np.array([[ 1, 0],[0, np.power(2,l-3) ]]),np.array([[1, -slk], [0, 1]]))
                i = np.mod(k, 2) + 2
                x[k] = nssfbrec( x[2*k], x[2*k+1], f1[i], f2[i], mkl )

        x[0] = nssfbrec( x[0], x[1], k1, k2, q1 )
        x[1] = nssfbrec( x[2], x[3], k1, k2, q1 )
        y = nssfbrec( x[0], x[1], k1, k2, [])
    return y

def nssfbrec( x1, x2, f1, f2, mup ):
    if mup == []:
        # y1 = imfilter( x1, f1 ,'symmetric')
        y1 = cv2.filter2D(x1, -1, f1, borderType=cv2.BORDER_REFLECT)
        # y2 = imfilter( x2, f2 ,'symmetric')
        y2 = cv2.filter2D(x2, -1, f2, borderType=cv2.BORDER_REFLECT)
        y = y1 + y2
        return y
    if mup.shape == (2, 2):
        y1 = myzconv2( x1, f1, mup )
        y2 = myzconv2( x2, f2, mup )
        y = y1 + y2
    return y

def NSPr(Ilow,Ihigh,g0,g1,level):
    level = level - 1
    if level != 0:
        index = []
        (m,n) = g0.shape
        Ng0= np.zeros((np.power(2,level) * m,np.power(2,level) * n))
        for i in range(0,Ng0.shape[0],np.power(2,level)):
            for j in range(0,Ng0.shape[1],np.power(2,level)):
                index.append([i,j])
        ind = 0
        for i in range(g0.shape[0]):
            for j in range(g0.shape[1]):
                Ng0[index[ind][0], index[ind][1]] = g0[i,j]
                ind += 1
        newg0 = Ng0[:(m-1)*np.power(2,level)+1,:(n-1)*np.power(2,level)+1]

        index = []
        (m,n) = g1.shape
        Ng1= np.zeros((np.power(2,level) * m,np.power(2,level) * n))
        for i in range(0,Ng1.shape[0],np.power(2,level)):
            for j in range(0,Ng1.shape[1],np.power(2,level)):
                index.append([i,j])
        ind = 0
        for i in range(g1.shape[0]):
            for j in range(g1.shape[1]):
                Ng1[index[ind][0], index[ind][1]] = g1[i,j]
                ind += 1
        newg1 = Ng1[:(m-1)*np.power(2,level)+1,:(n-1)*np.power(2,level)+1]

        Ired = cv2.filter2D(Ilow, -1, newg0, borderType=cv2.BORDER_REFLECT) + cv2.filter2D(Ihigh, -1, newg1, borderType=cv2.BORDER_REFLECT)
    else:
        Ired = cv2.filter2D(Ilow, -1, g0, borderType=cv2.BORDER_REFLECT) + cv2.filter2D(Ihigh, -1, g1, borderType=cv2.BORDER_REFLECT)
    return Ired


if __name__ == "__main__":
    I = cv2.imread('zoneplate.png', 0)
    I = np.array(I, dtype=np.float32)
    I = cv2.normalize(I, None, alpha=0, beta=1.0, norm_type=cv2.NORM_MINMAX)

    levels = [1, 2, 3, 4]
    pname = 'pyr'
    dname = 'pkva'
    type = 'NSCT'

    [Insp,Insct]= myNSCTd(I,levels,pname,dname,type)

    IMf = myNSCTr(Insct,levels,pname,dname,type)
    
    diff = np.abs(IMf - I)
    plt.figure('1')
    plt.subplot(121),plt.imshow(IMf),plt.title('IMf')
    plt.subplot(122),plt.imshow(I),plt.title('I')
    plt.figure('2')
    plt.subplot(111),plt.imshow(diff),plt.title('diff')
    plt.show()
