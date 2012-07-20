from cgp import *
from os.path import join as path_join
import os

title = 'Fast inverter2012-07-14-20'

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
.model 2N3906  PNP(Is=455.9E-18 Xti=3 Eg=1.11 Vaf=33.6 Bf=204 Ise=7.558f
+               Ne=1.536 Ikf=.3287 Nk=.9957 Xtb=1.5 Var=100 Br=3.72
+               Isc=529.3E-18 Nc=15.51 Ikr=11.1 Rc=.8508 Cjc=10.13p Mjc=.6993
+               Vjc=1.006 Fc=.5 Cje=10.39p Mje=.6931 Vje=.9937 Tr=10n Tf=181.2p
+               Itf=4.881m Xtf=.7939 Vtf=10 Rb=10, level=1)
.model 2N3904   NPN(Is=6.734f Xti=3 Eg=1.11 Vaf=74.03 Bf=416.7 Ne=1.259
+               Ise=6.734f Ikf=66.78m Xtb=1.5 Br=.7371 Nc=2 Isc=0 Ikr=0 Rc=1
+               Cjc=3.638p Mjc=.3085 Vjc=.75 Fc=.5 Cje=4.493p Mje=.2593 Vje=.75
+               Tr=239.5n Tf=301.2p Itf=.4 Vtf=4 Xtf=2 Rb=10, level=1)
Vin na 0
rin na n1 50
vc n0 0 5
rload n2 0 100k
""",
"""
.control
tran 1n 400n
print v(n2)
print i(vin)
print i(vc)
.endc
.model 2N3906  PNP(Is=455.9E-18 Xti=3 Eg=1.11 Vaf=33.6 Bf=204 Ise=7.558f
+               Ne=1.536 Ikf=.3287 Nk=.9957 Xtb=1.5 Var=100 Br=3.72
+               Isc=529.3E-18 Nc=15.51 Ikr=11.1 Rc=.8508 Cjc=10.13p Mjc=.6993
+               Vjc=1.006 Fc=.5 Cje=10.39p Mje=.6931 Vje=.9937 Tr=10n Tf=181.2p
+               Itf=4.881m Xtf=.7939 Vtf=10 Rb=10, level=1)
.model 2N3904   NPN(Is=6.734f Xti=3 Eg=1.11 Vaf=74.03 Bf=416.7 Ne=1.259
+               Ise=6.734f Ikf=66.78m Xtb=1.5 Br=.7371 Nc=2 Isc=0 Ikr=0 Rc=1
+               Cjc=3.638p Mjc=.3085 Vjc=.75 Fc=.5 Cje=4.493p Mje=.2593 Vje=.75
+               Tr=239.5n Tf=301.2p Itf=.4 Vtf=4 Xtf=2 Rb=10, level=1)
Vin na 0 PULSE(0 5 20n 1n 1n 1 1)
rin na n1 50
vc n0 0 5
rload n2 0 100k
""",
"""
.control
tran 1n 400n
print v(n2)
print i(vin)
print i(vc)
.endc
.model 2N3906  PNP(Is=455.9E-18 Xti=3 Eg=1.11 Vaf=33.6 Bf=204 Ise=7.558f
+               Ne=1.536 Ikf=.3287 Nk=.9957 Xtb=1.5 Var=100 Br=3.72
+               Isc=529.3E-18 Nc=15.51 Ikr=11.1 Rc=.8508 Cjc=10.13p Mjc=.6993
+               Vjc=1.006 Fc=.5 Cje=10.39p Mje=.6931 Vje=.9937 Tr=10n Tf=181.2p
+               Itf=4.881m Xtf=.7939 Vtf=10 Rb=10, level=1)
.model 2N3904   NPN(Is=6.734f Xti=3 Eg=1.11 Vaf=74.03 Bf=416.7 Ne=1.259
+               Ise=6.734f Ikf=66.78m Xtb=1.5 Br=.7371 Nc=2 Isc=0 Ikr=0 Rc=1
+               Cjc=3.638p Mjc=.3085 Vjc=.75 Fc=.5 Cje=4.493p Mje=.2593 Vje=.75
+               Tr=239.5n Tf=301.2p Itf=.4 Vtf=4 Xtf=2 Rb=10, level=1)
Vin na 0 PULSE(5 0 20n 1n 1n 1 1)
rin na n1 50
vc n0 0 5
rload n2 0 100k
"""
]

#Dictionary of the availabe parts
parts = {'R':{'nodes':2,'value':1,'min':0,'max':7},
         'C':{'nodes':2,'value':1,'min':-13,'max':-11},
         #'L':{'nodes':2,'value':1,'min':-9,'max':-3},
         #'D1':{'nodes':2,'spice':'1N4148'},
         'Q1':{'nodes':3,'spice':'2N3904'},
         'Q2':{'nodes':3,'spice':'2N3906'}
         }

def goalinv(f,k,**kwargs):
    """k is the name of measurement. eq. v(n2)"""
    #This is the DC transfer curve goal function
    if k=='v(n2)':
        #kwargs['extra'][0] is the transition voltage
        return 5 if f<=kwargs['extra'][0] else 0
    elif k[0]=='i':
        #Goal for current use is 0
        return 0

def transient_goal_inv(f,k,**kwargs):
    #Goal for first transient simulation
    n = 10**(-9)
    if k[0]=='v':
        if f<=20*n:
            return 5
        elif 20*n<f:
            return 0
    return 0#For current

def transient_goal_inv2(f,k,**kwargs):
    #Second transient simulation
    n = 10**(-9)
    if k[0]=='v':
        if f<=20*n:
            return 0
        elif 20*n<f:
            return 5
    return 0

def constraint0(f,x,k,**kwargs):
    #Constraint for static current consumption
    if k[0]=='i':
        return abs(x)<2e-3
    return True

def constraint1(f,x,k,**kwargs):
    """f is input, x is output, k is measurement, extra is extra value of chromosome"""
    #Constraint for the first transient simulation
    if k[0]=='v' and f<2e-9:
        #Output should be 0.3V above the transition voltage at t=0
        return x>kwargs['extra'][0]+0.3
    if k[0]=='v' and f>290e-9:
        #And below it after the transition on the input
        return x<kwargs['extra'][0]-0.3
    if k[0]=='i':
        #Goal for current use
        return abs(x)<5e-3
    return True

def constraint2(f,x,k,**kwargs):
    """f is input, x is output, k is measurement, extra is extra value of chromosome"""
    #Same as last one, but with other way around
    if k[0]=='v' and f<2e-9:
        return x<kwargs['extra'][0]-0.3
    if k[0]=='v' and f>290e-9:
        return x>kwargs['extra'][0]+0.3
    if k[0]=='i':
        return abs(x)<5e-3
    return True


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
        print 'No save file found'
        pass

    if resume:
        #Resuming from file
        resumed = pickle.load(r_file)
        outfile = open(resumed[2],'a')
    else:
        outfile = open(path_join(title,'sim'+strftime("%Y-%m-%d %H:%M:%S")+'.log'),'w')

    e = CGP(pool_size=4000,
            nodes=8,
            parts_list=parts,
            max_parts=12,#Maximum number of parts
            elitism=1,#Best circuit is copied straight to next generation
            mutation_rate=0.7,
            crossover_rate=0.2,
            fitnessfunction=[goalinv,transient_goal_inv,transient_goal_inv2],
            fitness_weight=[{'v(n2)':10,'i(vc)':5000,'i(vin)':5000},{'v(n2)':2,'i(vc)':3000,'i(vin)':3000},{'v(n2)':2,'i(vc)':3000,'i(vin)':3000}],#Current use is weighted very heavily
            extra_value=[(0.5,4.5)],#This is used as transition value
            constraints=[constraint0,constraint1,constraint2],
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
