import matplotlib.pyplot as plt
import subprocess
import math
import sys


def bode_plot(v,phase=None,**kwargs):
    plt.figure()
    for i,k in enumerate(v.keys(),1):
        freq=v[k][0]
        gain=v[k][1]
        if phase!=None:
            plt.subplot(211)

        plt.subplot(str(i)+'11')
        # plot it as a log-scaled graph
        plt.plot(freq,gain,label=k)
        #plt.semilogx(freq,gain,basex=10,**kwargs)

        # update axis ranges
        ax = []
        ax[0:4] = plt.axis()
        # check if we were given a frequency range for the plot
        plt.axis(ax)

        plt.grid(True)
        # turn on the minor gridlines to give that awesome log-scaled look
        plt.grid(True,which='minor')
        plt.ylabel("Gain (dB)")

        if phase!=None:
            plt.subplot(212)
            plt.semilogx(freq, phase,basex=10,**kwargs)

            # update axis ranges, we know the phase is between -pi and pi
            ax = plt.axis()
            plt.axis([ax[0],ax[1],-math.pi,math.pi])

            plt.grid(True)
            plt.grid(True,which='minor')
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Phase (rads)")

            # nice LaTeX pi scale for the phase part of the plot
            plt.yticks((-math.pi,-math.pi/2,0,math.pi/2,math.pi),
                       (r"$-\pi$",r"$-\frac{\pi}{2}$","0",r"$\frac{\pi}{2}$",r"$\pi$"))
    plt.legend(v.keys())
    plt.show()

def parse_output(output,values,start=1):
    value=[[] for i in xrange(values)]
    output=output.split('\n')
    n=0
    for line in xrange(n,len(output)):
        temp=output[line].replace(',','').split()
        if len(temp)>2:
            try:
                #time.append(float(temp[1]))
                #value.append(float(temp[2]))
                for i in xrange(values):
                    value[i].append(float(temp[start+i]))
            except ValueError:
                continue
    return value

def parse_output2(output):
    value={}
    output=output.split('\n')
    index=1
    current = ()
    for line in xrange(len(output)):
        temp=output[line].replace(',','').split()
        if len(temp)>0:
            if temp[0]=='Index':
                temp2=output[line+2].replace(',','').split()
                if float(temp2[0])<index:
                    current = temp[2]
                    value[temp[2]]=([],[])
                    index=0

        if len(temp)>2:
            try:
                float(temp[1]),float(temp[2])
            except:
                continue
            index+=1
            for i in xrange(2):
                value[current][0].append(float(temp[1]))
                value[current][1].append(float(temp[2]))
    return value

def simulate(file):
        spice = subprocess.Popen(['ngspice', '-s'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        output = spice.communicate(file)[0]
        return parse_output2(output)


w="""
.control
$ac dec 30 1 100khz
dc V1 0 10 0.05
print i(V1)
.endc
$.MODEL QMOD NPN(BF=100 CJC=20pf CJE=20pf IS=1E-16)
$.MODEL QMOD2 PNP(BF=100 CJC=20pf CJE=20pf IS=1E-16)
.model 2N3906  PNP(Is=455.9E-18 Xti=3 Eg=1.11 Vaf=33.6 Bf=204.7 Ise=7.558f
+               Ne=1.536 Ikf=.3287 Nk=.9957 Xtb=1.5 Var=100 Br=3.72
+               Isc=529.3E-18 Nc=15.51 Ikr=11.1 Rc=.8508 Cjc=10.13p Mjc=.6993
+               Vjc=1.006 Fc=.5 Cje=10.39p Mje=.6931 Vje=.9937 Tr=10n Tf=181.2p
+               Itf=4.881m Xtf=.7939 Vtf=10 Rb=10)
*
*               Fairchild  pid=66   case=TO92
*               11/19/2001 calccb update
*$
.model 2N3904   NPN(Is=6.734f Xti=3 Eg=1.11 Vaf=74.03 Bf=416.4 Ne=1.259
+               Ise=6.734f Ikf=66.78m Xtb=1.5 Br=.7371 Nc=2 Isc=0 Ikr=0 Rc=1
+               Cjc=3.638p Mjc=.3085 Vjc=.75 Fc=.5 Cje=4.493p Mje=.2593 Vje=.75
+               Tr=239.5n Tf=301.2p Itf=.4 Vtf=4 Xtf=2 Rb=10)
*               Fairchild        pid=23          case=TO92
*               88-09-08 bam    creation
$Vin n1 0 dc 0 ac 1
V1 n1 0

Q289250128 0 n4 n7 2N3904
Q289299352 n6 0 n4 2N3904
Q1273979728 n6 0 n4 2N3906
Q158204240 n8 n9 n2 2N3906
Q270367352 n1 n2 n8 2N3904
R273979872 0 n0 7390000.56192
Q2129392152 n3 n8 n7 2N3904
Q2169627088 n4 0 n6 2N3904
"""

output = simulate(w)
bode_plot(output)

