from cgp import *
from os.path import join as path_join
import os

title = 'Fast inverter2012-07-14-15'

if not os.path.exists(title):
    os.makedirs(title)

options=[
"""
.control
dc Vin 0 5 0.1
print v(n2)
print i(Vin)
print i(vc)
.endc
$.MODEL QMOD NPN(BF=100 CJC=20pf CJE=20pf IS=1E-16)
$.MODEL QMOD2 PNP(BF=100 CJC=20pf CJE=20pf IS=1E-16)
.model 2N3906  PNP(Is=455.9E-18 Xti=3 Eg=1.11 Vaf=33.6 Bf=204 Ise=7.558f
+               Ne=1.536 Ikf=.3287 Nk=.9957 Xtb=1.5 Var=100 Br=3.72
+               Isc=529.3E-18 Nc=15.51 Ikr=11.1 Rc=.8508 Cjc=10.13p Mjc=.6993
+               Vjc=1.006 Fc=.5 Cje=10.39p Mje=.6931 Vje=.9937 Tr=10n Tf=181.2p
+               Itf=4.881m Xtf=.7939 Vtf=10 Rb=10, level=1)
*
*               Fairchild  pid=66   case=TO92
*               11/19/2001 calccb update
*$
.model 2N3904   NPN(Is=6.734f Xti=3 Eg=1.11 Vaf=74.03 Bf=416.7 Ne=1.259
+               Ise=6.734f Ikf=66.78m Xtb=1.5 Br=.7371 Nc=2 Isc=0 Ikr=0 Rc=1
+               Cjc=3.638p Mjc=.3085 Vjc=.75 Fc=.5 Cje=4.493p Mje=.2593 Vje=.75
+               Tr=239.5n Tf=301.2p Itf=.4 Vtf=4 Xtf=2 Rb=10, level=1)
*               Fairchild        pid=23          case=TO92
*               88-09-08 bam    creation
.MODEL 1N4148 D
+ IS = 4.352E-9
+ N = 1.906
+ BV = 110
+ IBV = 0.0001
+ RS = 0.6458
+ CJO = 7.048E-13
+ VJ = 0.869
+ M = 0.03
+ FC = 0.5
+ TT = 3.48E-9
Vin na 0
rin na n1 1k
vc n0 0 5
rload n2 0 1k
""",
"""
.control
tran 0.5n 50n
print v(n2)
.endc
$.MODEL QMOD NPN(BF=100 CJC=20pf CJE=20pf IS=1E-16)
$.MODEL QMOD2 PNP(BF=100 CJC=20pf CJE=20pf IS=1E-16)
.model 2N3906  PNP(Is=455.9E-18 Xti=3 Eg=1.11 Vaf=33.6 Bf=204 Ise=7.558f
+               Ne=1.536 Ikf=.3287 Nk=.9957 Xtb=1.5 Var=100 Br=3.72
+               Isc=529.3E-18 Nc=15.51 Ikr=11.1 Rc=.8508 Cjc=10.13p Mjc=.6993
+               Vjc=1.006 Fc=.5 Cje=10.39p Mje=.6931 Vje=.9937 Tr=10n Tf=181.2p
+               Itf=4.881m Xtf=.7939 Vtf=10 Rb=10, level=1)
*
*               Fairchild  pid=66   case=TO92
*               11/19/2001 calccb update
*$
.model 2N3904   NPN(Is=6.734f Xti=3 Eg=1.11 Vaf=74.03 Bf=416.7 Ne=1.259
+               Ise=6.734f Ikf=66.78m Xtb=1.5 Br=.7371 Nc=2 Isc=0 Ikr=0 Rc=1
+               Cjc=3.638p Mjc=.3085 Vjc=.75 Fc=.5 Cje=4.493p Mje=.2593 Vje=.75
+               Tr=239.5n Tf=301.2p Itf=.4 Vtf=4 Xtf=2 Rb=10, level=1)
*               Fairchild        pid=23          case=TO92
*               88-09-08 bam    creation
.MODEL 1N4148 D
+ IS = 4.352E-9
+ N = 1.906
+ BV = 110
+ IBV = 0.0001
+ RS = 0.6458
+ CJO = 7.048E-13
+ VJ = 0.869
+ M = 0.03
+ FC = 0.5
+ TT = 3.48E-9
Vin na 0 PULSE(0 5 20n 1n 1n 1 1)
rin na n1 1k
vc n0 0 5
rload n2 0 1k
""",
"""
.control
tran 0.5n 50n
print v(n2)
.endc
$.MODEL QMOD NPN(BF=100 CJC=20pf CJE=20pf IS=1E-16)
$.MODEL QMOD2 PNP(BF=100 CJC=20pf CJE=20pf IS=1E-16)
.model 2N3906  PNP(Is=455.9E-18 Xti=3 Eg=1.11 Vaf=33.6 Bf=204 Ise=7.558f
+               Ne=1.536 Ikf=.3287 Nk=.9957 Xtb=1.5 Var=100 Br=3.72
+               Isc=529.3E-18 Nc=15.51 Ikr=11.1 Rc=.8508 Cjc=10.13p Mjc=.6993
+               Vjc=1.006 Fc=.5 Cje=10.39p Mje=.6931 Vje=.9937 Tr=10n Tf=181.2p
+               Itf=4.881m Xtf=.7939 Vtf=10 Rb=10, level=1)
*
*               Fairchild  pid=66   case=TO92
*               11/19/2001 calccb update
*$
.model 2N3904   NPN(Is=6.734f Xti=3 Eg=1.11 Vaf=74.03 Bf=416.7 Ne=1.259
+               Ise=6.734f Ikf=66.78m Xtb=1.5 Br=.7371 Nc=2 Isc=0 Ikr=0 Rc=1
+               Cjc=3.638p Mjc=.3085 Vjc=.75 Fc=.5 Cje=4.493p Mje=.2593 Vje=.75
+               Tr=239.5n Tf=301.2p Itf=.4 Vtf=4 Xtf=2 Rb=10, level=1)
*               Fairchild        pid=23          case=TO92
*               88-09-08 bam    creation
.MODEL 1N4148 D
+ IS = 4.352E-9
+ N = 1.906
+ BV = 110
+ IBV = 0.0001
+ RS = 0.6458
+ CJO = 7.048E-13
+ VJ = 0.869
+ M = 0.03
+ FC = 0.5
+ TT = 3.48E-9
Vin na 0 PULSE(5 0 20n 1n 1n 1 1)
rin na n1 1k
vc n0 0 5
rload n2 0 1k
"""
]

#Dictionary of the availabe parts
parts = {'R':{'nodes':2,'value':1,'min':0,'max':7},
         'C':{'nodes':2,'value':1,'min':-13,'max':-4},
         #'L':{'nodes':2,'value':1,'min':-9,'max':-3},
         #'D1':{'nodes':2,'spice':'1N4148'},
         'Q1':{'nodes':3,'spice':'2N3906'},
         'Q2':{'nodes':3,'spice':'2N3904'}
         }

def goalinv(f,k,extra):
    """k is the name of measurement. eq. v(n2)"""
    if k=='v(n2)':
        return 5 if f<=extra else 0
    elif k[0]=='i':
        return 0

def transient_goal_inv(f,k,extra):
    n = 10**(-9)
    if f<=20*n:
        return 5
    elif 20*n<f:
        return 0

def transient_goal_inv2(f,k,extra):
    n = 10**(-9)
    if f<=20*n:
        return 0
    elif 20*n<f:
        return 5


if __name__ == "__main__":
    resume = False
    try:
        r_file = open(path_join(title,'.dump'),'r')
        print "Do you want to resume Y/n:"
        while True:
            r = getch._Getch()()
            if r=='':
                resume = True
                print "Resumed"
                break
            if r in ('y','Y','\n'):
                resume = True
                print "Resumed"
                break
            if r in ('n','N'):
                break
    except IOError:
        pass

    if resume:
        #Resuming from file
        resumed = pickle.load(r_file)
        outfile = open(resumed[2],'a')
    else:
        outfile = open(path_join(title,'sim'+strftime("%Y-%m-%d %H:%M:%S")+'.log'),'w')

    e = CGP(pool_size=4000,
            nodes=16,
            parts_list=parts,
            max_parts=18,
            elitism=1,
            mutation_rate=0.7,
            crossover_rate=0.2,
            fitnessfunction=[goalinv,transient_goal_inv,transient_goal_inv2],
            fitness_weight=[{'v(n2)':10,'i(vc)':20,'i(vin)':20},{'v(n2)':2},{'v(n2)':2}],
            extra_value=(0.7,4.3),#This is used as transition value
            constraints=[None,None,None],
            spice_sim_commands=options,
            log=outfile,
            directory=title,
            resumed=resume,
            plot_titles=[{'v(n2)':"DC sweep",'i(vc)':"Current from power supply",'i(vin)':'Current from logic input'},{'v(n2)':'Step response'},{'v(n2)':'Step response'}],
            plot_yrange={'v(n2)':(0,5),'i(vin)':(-0.2,0.02),'i(vc)':(-0.2,0.02)})

    if resume:
        e.generation = resumed[0]
        e.pool = resumed[1]
    e.run()
