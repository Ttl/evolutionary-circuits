#TTL NAND-gate optimization
#Very slow simulation and usually takes somewhere close to 30-40 generations
#to find a working gate
title = 'nand'

#Put SPICE device models used in the simulations here.
models="""
.model 2N3906  PNP(Is=455.9E-18 Xti=3 Eg=1.11 Vaf=33.6 Bf=204 Ise=7.558f
+               Ne=1.536 Ikf=.3287 Nk=.9957 Xtb=1.5 Var=100 Br=3.72
+               Isc=529.3E-18 Nc=15.51 Ikr=11.1 Rc=.8508 Cjc=10.13p Mjc=.6993
+               Vjc=1.006 Fc=.5 Cje=10.39p Mje=.6931 Vje=.9937 Tr=10n Tf=181.2p
+               Itf=4.881m Xtf=.7939 Vtf=10 Rb=10, level=1)
.model 2N3904   NPN(Is=6.734f Xti=3 Eg=1.11 Vaf=74.03 Bf=416.7 Ne=1.259
+               Ise=6.734f Ikf=66.78m Xtb=1.5 Br=.7371 Nc=2 Isc=0 Ikr=0 Rc=1
+               Cjc=3.638p Mjc=.3085 Vjc=.75 Fc=.5 Cje=4.493p Mje=.2593 Vje=.75
+               Tr=239.5n Tf=301.2p Itf=.4 Vtf=4 Xtf=2 Rb=10, level=1)
"""

#5V power supply with series resistance of 10 ohms.
#Bypass capacitor with series resistance of 0.1 ohms.
#10k ohm and 100pF of load
common="""
Vc na 0 5
Rc na vc 10
cv na nb 10n
rcv nb vc 100m
rload out 0 10k
cload out 0 100p
"""

inputs = ['in1','in2']
outputs = ['out']
special_nodes = ['vc']

#List of simulations
#Put SPICE simulation options between.control and .endc lines
spice_commands=[
#Functionality
"""
.control
tran 5n 100u
print v(out)
print i(vc)
print i(Vpwl1)
print i(Vpwl2)
.endc
Vpwl1 in1 0 0 PWL(0 0 20u 0 20.05u 5 40u 5 40.05u 0 50u 0 50.05u 5 60u 5 60.05u 0 70u 0 70.05u 5)
Vpwl2 in2 0 0 PWL(0 0 10u 0 10.05u 5 20u 5 20.05u 0 30u 0 30.05u 5 40u 5 40.05u 0 60u 0 60.05u 5)
""",
#Input isolation test 1
"""
.control
tran 10u 20u
print v(in1)
.endc
Vin in2 0 0 PWL(0 0 5u 0 15u 5 20u 5)
rin in1 0 100k
"""
,
#Input isolation test 2
"""
.control
tran 10u 20u
print v(in2)
.endc
Vin in1 0 0 PWL(0 0 5u 0 15u 5 20u 5)
rin in2 0 100k
"""

]

#Dictionary of the availabe parts
parts = {'R':{'nodes':2,'value':(1,1e6)},#Resistors
         #'C':{'nodes':2,'value':(1e-12,1e-7)},#Capacitors
         #'L':{'nodes':2,'value':(1e-10,1e-5)},#Inductors
         'Q':{'nodes':3,'model':('2N3904','2N3906')},#NPN/PNP transistors
         }

def _goal(f,k,**kwargs):
    """k is the name of measurement. eq. v(out)"""
    #Functionality
    if k=='v(out)':
        if (30.05e-6<f<40e-6) or (f>70.05e-6):
            return 0
        return kwargs['extra'][0]
    #Current
    elif k[0]=='i':
        #Goal for current use is 0
        return 0
    #Input isolation
    elif k in ('v(in1)','v(in2)'):
        return 0

def _constraint0(f,x,k,**kwargs):
    if k[0] == 'v':
        if (32e-6<f<38e-6) or (f>72e-6):
            return x<1
        if f<20e-6 or (45e-6<f<59e-6) or (65e-6<f<69e-6) or (22e-6<f<29e-6):
            return kwargs['extra'][0]+0.1>x>kwargs['extra'][0]-0.1
    return True

def _weight(x,**kwargs):
    """Weighting function for scoring"""
    #Low weight when glitches are allowed
    if abs(x-20e-6)<8e-6:
        return 0.004
    if abs(x-60e-6)<8e-6:
        return 0.004
    #High weight on the edges
    if 0<x-40e-6<5e-6:
        return 5.0
    if 0<x-70e-6<5e-6:
        return 5.0
    return 0.07

##TTL npn NAND-gate seed circuit
#seed="""
#R1 n4 vc 1k
#R2 out vc 1k
#Q11 n5 n4 in1 2N3904
#Q12 n5 n4 in2 2N3904
#Q13 out n5 0 2N3904
#"""
#seed_copies = 300

#Default timeout is too low
timeout=2.5
population=2000#Too small population might not converge, but is faster to simulate
max_parts=10#Maximum number of parts
elitism=1#Best circuit is copied straight to next generation, default setting
constraints = [_constraint0,None,None]

mutation_rate=0.70
crossover_rate=0.10

plot_every_generation = True
fitness_function=[_goal]*3
fitness_weight=[{'v(out)':_weight,'i(vc)':3000,'i(vpwl1)':1000,'i(vpwl2)':1000},{'v(in1)':0.05},{'v(in2)':0.05}]
#On state output voltage
extra_value=[(4.5,5.0)]

plot_yrange={'v(out)':(-0.5,5.5),'i(vin)':(-0.1,0.01),'i(vc)':(-0.1,0.01),'v(in1)':(-0.5,5.5),'v(in2)':(-0.5,5.5)}
