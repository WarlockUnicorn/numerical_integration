# =============================================================================
#   ***** monte_carlo_integration.py *****
#   Python script to solve and plot Marshak Waves (n=0 & n=3).
#
#   Author:     Ryan Clement
#   Created:    October 2021
#
#   Change Log:
#   Who:
#   Date:       MM/DD/YYY
#   What:
#
#   Who:
#   Date:       MM/DD/YYYY
#   What:
# =============================================================================

# Imports
import numpy as np
from scipy.stats import norm
from scipy.stats import sem as error
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection

rng = np.random.default_rng()

def genFloats(a=0,b=1,n=10):
    return a + (b-a)*rng.random(n)

func = lambda x: 1.0 + x*x

def uniform_integrate(func,a,b,n=1000):
    """ Simple Monte Carlo Quadrature"""
    x = genFloats(a,b,n)
    y = func(x)
    est = (b-a)*(y.sum()/n)
    return est

def geometric_integrate(func,a,b,n=1000):
    """ Function maximum must be on end-points."""
    aF = func(a)
    bF = func(b)
    maxF = max(aF,bF)
    x = genFloats(a,b,n)
    y = func(x)
    h = genFloats(0,maxF,n)
    res = (b-a)*maxF*(h <= y).sum()/n
    print(f'Geo Area = {res}')
    return res    

# geometric_integrate(func,0,4,1000)

def geometric_integrate_plot(func,a,b,n=1000):
    """ Function maximum must be on end-points."""
    aF = func(a)
    bF = func(b)
    maxF = max(aF,bF)
    x = genFloats(a,b,n)
    y = func(x)
    h = genFloats(0,maxF,n)
    xIn = x[h <= func(x)]
    xOut = x[h > func(x)]
    hIn = h[h <= y]
    hOut = h[h > y]
    res = (b-a)*maxF*(h <= y).sum()/n
    xP = np.linspace(0,4,100)
    yP = func(xP)
    print('Geo Area = ',res)
    plt.scatter(xIn,hIn,color='red')
    plt.scatter(xOut,hOut,color='blue')
    plt.plot(xP,yP,color='black',linewidth=5)
    plt.xlim(0,4)
    plt.ylim(0,bF)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('geoMC_integrate.png')
    plt.show()
    return res  


def uniform_integrate_sub(func,a,b,n=1000,s=10):
    """ Effectively, using a grid! Why not use quadrature?"""
    dn = n//s
    xz = []
    yz = []
    dx = (b-a)/s
    sum = 0.0
    for i in range(s):
        minI = a + i*dx
        maxI = a + (i+1)*dx
        fArr = func(genFloats(minI,maxI,dn))
        sum += fArr.sum()
    est = (b-a)*sum/n
    return est

def show_rects(func,a,b,n=10):
    fig, ax = plt.subplots()
    xArr = genFloats(a,b,n)
    xw = (b-a)/float(n)
    shift = xw/2.0
    yhArr = func(xArr)
    rects = [Rectangle((x-shift,0),xw,y) for x, y in zip(xArr,yhArr)]
    pc = PatchCollection(rects,facecolor='blue',edgecolor='black',alpha=0.3)
    ax.add_collection(pc)
    xF = np.linspace(0,4,100)
    yF = func(xF)
    ax.plot(xF,yF,color='black',label='Parabola')
    ax.text(0,15,f'N={n} Samples')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(r'Parabola: $y = 1 + x^2$')
    ax.legend()
    # fig.savefig('monte_carlo_rects.png')
    plt.show()

def plot_uniform_dist():
    rflts = genFloats(1,10,1000)
    bns = np.arange(1,11)
    plt.hist(rflts,bins=bns,density=True,color='blue',histtype='stepfilled')
    plt.plot([1,10],2*[1.0/9.0],color='red')
    plt.xlabel('Bins')
    plt.ylabel('Density')
    plt.title('Uniform PDF on [1,10) for n=1000')
    # plt.text(5,0.12,'n=1000')
    plt.grid(True)
    plt.xlim(1,10)
    plt.ylim(0,0.15)
    # plt.savefig('uniform_distribution.png')
    plt.show()

def anal_runs():
    n = 10000
    areas = np.zeros(n)
    numBins = 200
    analA = 25.333333333
    for i in range(n):
        areas[i] = uniform_integrate(func,0,4,10000)
    mu, std = norm.fit(areas)
    err = std/np.sqrt(n)
    print(f'Mean = {mu}')
    print(f'Standard Deviation = {std}')
    print(f'Error = {err}')
    plt.hist(areas, numBins, density=True, color='blue',alpha=0.8,label='Bins')
    xmin, xmax = plt.xlim()
    xNorm = np.linspace(xmin,xmax,100)
    yNorm = norm.pdf(xNorm,mu,std)
    plt.plot(xNorm,yNorm,color='black',label='Normal Distribution')
    plt.plot([analA,analA],[0,1],color='red',label="Exact")
    plt.xlabel('Integration Result')
    plt.ylabel('Density')
    plt.title(f'Distribution of {n:,} Runs')
    plt.legend()
    # plt.savefig('monte_carlo_anal.png')
    plt.show()

def err_anal():
    nums = np.array([100,1000,10000,100000,1000000])
    errs = np.zeros(5)
    i = 0
    a = np.zeros(1000)
    js = np.arange(1000)
    for n in nums:
        for j in js:
            a[j] = uniform_integrate(func,0,4,n)
        errs[i] = error(a)
        i+=1
    figAE, axAE = plt.subplots()
    axAE.plot(nums,errs,'ro-',label='Estimate')
    axAE.plot(nums,1/np.sqrt(nums),'bo-',label=r'$1/\sqrt{N}$')
    axAE.set_xscale('log')
    axAE.set_yscale('log')
    axAE.grid(True)
    axAE.set_xlabel(r'$N$')
    axAE.set_ylabel('Error')
    axAE.set_ylim(5e-4,1.3e-1)
    axAE.legend()
    # plt.savefig('monte_carlo_error.png')
    plt.show()

def box(c,w,h,xA):
    yA = np.zeros(xA.size)
    i=0
    hw = w/2.0
    a = c - hw
    b = c + hw
    for x in xA:
        if x>=a:
            if x<=b:
                yA[i] = h
            else:
                yA[i] = 0
        else:
            yA[i] = 0
        i+=1
    return yA

def plot_box(a,b,h,xA):
    figB, axB = plt.subplots()
    yA = np.zeros(xA.size)
    w = b - a
    c = a + w/2.0
    yA = box(c,w,h,xA)
    axB.plot(xA,yA,'b-',label='box')
    axB.legend()
    plt.show()

def mc_box_convolve(a,b,n,func):
    """Convolve func with unit box on [a,b]."""
    ep = 101    # Number of evaluation points for convolution
    tA = np.linspace(a,b,ep)
    conv = np.zeros(ep)
    i=0
    for t in tA:
        xA = genFloats(t-0.5,t+0.5,n)
        yA = func(xA)
        conv[i] = yA.sum()/n
        i+=1
    plt.plot(tA,conv,'r-',label='Convolution')
    plt.legend()
    plt.xlabel("x")
    plt.show()



if __name__ == '__main__':
    # show_rects(func,0,4,33)
    # plot_uniform_dist()
    # exactA = 25.333333333
    # res = uniform_integrate(func,0,4,1000)
    # print(f'Area = {res}')
    # diff = 100*np.abs((exactA-res)/exactA)
    # print('Diff = ',diff)
    # resS = uniform_integrate_sub(func,0,4,1000,20)
    # print(f'Area S = {resS}')
    # diffS = 100*np.abs((exactA-resS)/exactA)
    # print('Diff S = ',diffS)
    # geometric_integrate_plot(func,0,4,1000)
    anal_runs()
    err_anal()
    x = np.linspace(3,7,100)
    plot_box(4.5,5.5,1,x)
    mc_box_convolve(3,7,100000,lambda x: box(5,1,1,x))
