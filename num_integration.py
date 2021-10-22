# =============================================================================
#  ***** num_integration.py *****
#  Python script to plot numerical integration information.
#
#  Author:     Ryan Clement
#  Created:    July 2021
#
#  Change Log:
#  Who:
#  Date:       MM/DD/YYY
#  What:
#
#  Who:
#  Date:       MM/DD/YYYY
#  What:
# =============================================================================

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle


def parabola(x):
    return 1.0 + x**2

def solveExact(a,b):
    result = (3*b + b**3 - 3.0*a - a**3)/3.0
    print(f"Exact: {result:.5f}")
    return result

def solveLeftRectangleRule(a,b,n):
    dx = (b-a)/(n-1)
    xPoints = np.linspace(a,b-dx,n-1)
    yPoints = parabola(xPoints)
    result = dx*np.sum(yPoints)
    print(f'Left end-point rectangle rule (LR): {result:.5f}')
    return result

def solveRightRectangleRule(a,b,n):
    dx = (b-a)/(n-1)
    xPoints = np.linspace(a+dx,b,n-1)
    yPoints = parabola(xPoints)
    result = dx*np.sum(yPoints)
    print(f'Right end-point rectangle rule (RR): {result:.5f}')
    return result
    
def solveTrapezoidRule(a,b,n):
    dx = (b-a)/(2.0*(n-1))
    xPoints = np.linspace(a,b,n)
    yPoints = parabola(xPoints)
    result = dx*np.sum(yPoints[:-1] + yPoints[1:])
    print(f'Trapezoid rule (TR): {result:.5f}')
    return result
    
def solveSimpsonsRule(a,b,n):
    dx = (b-a)/(3.0*(n-1))
    x = np.linspace(a,b,n)
    y = np.array([1.0 + (i)**2 for i in x])
    result = dx*(y[0]+y[-1]+4.0*np.sum(y[1:-1:2])+2.0*np.sum(y[2:-1:2]))
    print(f"Simpson's rule (SR): {result:.5f}")
    return result

def plotLeftRectangleRule(a,b,n):
    dx = (b-a)/(n-1)
    xPoints = np.linspace(a,b,n)
    yPoints = parabola(xPoints)
    xSmooth = np.linspace(a,b,10*(n))
    ySmooth = parabola(xSmooth)
    # Plot Left Rect.
    figLR, axLR = plt.subplots()
    axLR.set_title(r'Parabola: $y = 1 + x^2$')
    axLR.set_xlabel('x')
    axLR.set_ylabel('y')
    axLR.set_xticks(xPoints)
    axLR.fill_between(xSmooth,0,ySmooth,color='salmon')
    for i in range(n-1):
        axLR.add_patch(Rectangle((xPoints[i],0),dx,yPoints[i],
                      edgecolor='navy',
                      facecolor='cornflowerblue'))
    axLR.plot(xPoints, yPoints, 'o', color='red')
    axLR.plot(xSmooth, ySmooth, color='red')
    # figLR.savefig('parabola_LeftRect.png')

def plotRightRectangleRule(a,b,n):
    dx = (b-a)/(n-1)
    xPoints = np.linspace(a,b,n)
    yPoints = parabola(xPoints)
    xSmooth = np.linspace(a,b,10*n)
    ySmooth = parabola(xSmooth)
    figRR, axRR = plt.subplots()
    axRR.set_title(r'Parabola: $y = 1 + x^2$')
    axRR.set_xlabel('x')
    axRR.set_ylabel('y')
    axRR.set_xticks(xPoints)
    for i in range(n-1):
        axRR.add_patch(Rectangle((xPoints[i],0),dx,yPoints[i+1],
                      edgecolor='navy',
                      facecolor='cornflowerblue'))
    axRR.plot(xPoints, yPoints, 'o', color='red')
    axRR.plot(xSmooth, ySmooth, color='red')
    # figRR.savefig('parabola_RightRect.png')

def plotTrapezoidRule(a,b,n):
    xPoints = np.linspace(a,b,n)
    yPoints = parabola(xPoints)
    xSmooth = np.linspace(a,b,10*n)
    ySmooth = parabola(xSmooth)
    figT, axT = plt.subplots()
    axT.set_title(r'Parabola: $y = 1 + x^2$')
    axT.set_xlabel('x')
    axT.set_ylabel('y')
    axT.set_xticks(xPoints)
    for i in range(n-1):
        axT.fill_between(xPoints,0,yPoints,color='cornflowerblue')
        axT.plot(xPoints[i:i+2],yPoints[i:i+2],color='navy')
        axT.plot((xPoints[i],xPoints[i]),(0,yPoints[i]),color='navy')
    axT.plot((xPoints[-1],xPoints[-1]),(0,yPoints[-1]),color='navy')
    axT.plot(xPoints, yPoints, 'o', color='red')
    axT.plot(xSmooth, ySmooth, color='red')
    # figT.savefig('parabola_trapezoid.png')


if __name__ == '__main__':
    a = 0.0 # Left Boundary
    b = 4.0 # Right Boundary
    n = 5   # Number of points
    plotLeftRectangleRule(a,b,n)
    plotRightRectangleRule(a,b,n)
    plotTrapezoidRule(a,b,n)
    ansLR = solveLeftRectangleRule(a,b,n)
    ansRR = solveRightRectangleRule(a,b,n)
    ansTR = solveTrapezoidRule(a,b,n)
    ansSR = solveSimpsonsRule(a,b,n)
    ansE  = solveExact(a,b)
    print('LR % err: {:.5f}%'.format(abs((ansLR-ansE)/ansE)*100))
    print('RR % err: {:.5f}%'.format(abs((ansRR-ansE)/ansE)*100))
    print('TR % err: {:.5f}%'.format(abs((ansTR-ansE)/ansE)*100))
    print('SR % err: {:.5f}%'.format(abs((ansSR-ansE)/ansE)*100))

