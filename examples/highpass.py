"""This file tries to evolve an active high-pass filter with cutoff frequency of 
1kHz. Devices allowed are: resistors, capacitors and TL071 opamps.

The reason there are so many different simulations is to make sure that load
resistance doesn't change the frequency response too much, make sure that filter
isn't unstable. Last simulation is to make sure that filter doesn't oscillate
without input signal.

A decent filter is usually found around generations 20-60.

Requires ngspice version 24 or newer compiled with xspice support.

TODO: Add a simulation to measure the output signal distortion.
"""
from math import log

title = 'High-pass_filter'

#Put SPICE device models used in the simulations here.
models="""
*                + - out vcc vee
.SUBCKT TL071    1 2  5   3   4
  C1   11 12 3.498E-12
  C2    6  7 15.00E-12
  DC    5 53 DX
  DE   54  5 DX
  DLP  90 91 DX
  DLN  92 90 DX
  DP    4  3 DX
  EGND 99 0 VALUE={0.5*v(3)+0.5*v(4)}
  GB 7 99 VALUE={4.715e6*i(VB)+5e6*(-i(VC)+i(VE)+i(VLP)-i(VLN))}
  GA    6  0 11 12 282.8E-6
  GCM   0  6 10 99 8.942E-9
  ISS   3 10 DC 195.0E-6
  HLIM 90  0 VLIM 1K
  J1   11  2 10 JX
  J2   12  1 10 JX
  R2    6  9 100.0E3
  RD1   4 11 3.536E3
  RD2   4 12 3.536E3
  RO1   8  5 150
  RO2   7 99 150
  RP    3  4 2.143E3
  RSS  10 99 1.026E6
  VB    9  0 DC 0
  VC    3 53 DC 2.200
  VE   54  4 DC 2.200
  VLIM  7  8 DC 0
  VLP  91  0 DC 25
  VLN   0 92 DC 25
.MODEL DX D(IS=800.0E-18)
.MODEL JX PJF(IS=15.00E-12 BETA=270.1E-6 VTO=-1)
.ENDS
"""

#This is for constructions common for every simulation
common="""
Vc ncc 0 10
Rc ncc nc 5
Ve nee 0 -10
Re nee ne 5
C1 nc ne 100n
"""

#List of simulations
#Put SPICE simulation options between.control and .endc lines
#
spice_commands=[
"""
.control
ac dec 30 10 1e5
print vdb(n2)
option rshunt = 1e12
.endc
Vin n1 0 ac 1
Rload n2 0 {aunif(200,100)}
""",
"""
.control
ac dec 30 10 1e5
print vdb(n2)
option rshunt = 1e12
.endc
Vin n1 0 ac 1
Rload n2 0 {aunif(50k,40k)}
""",
"""
.control
tran 50u 100m
print v(n2)
print i(vin)
print i(vc)
print i(ve)
option rshunt = 1e12
.endc
Vin n1 0 SIN(0 1 2k 0 0 0)
Rload n2 0 1k
""",
"""
.control
tran 100u 100m
print v(n2)
print i(vin)
print i(vc)
print i(ve)
option rshunt = 1e12
.endc
Vin n1 0 SIN(0 1 100 0 0 0)
Rload n2 0 1k
""",
"""
.control
tran 50u 100m
print v(n2)
print i(vin)
print i(vc)
print i(ve)
option rshunt = 1e12
.endc
Vin n1 0 0
Rload n2 0 1k
"""
]

#Dictionary of the availabe parts
#nodes: number of terminals
#value: Device has a value, eg. resistance, inductance
#min and max: Range for value.
#spice: This is added after anything else, can be used to specify device model
#or some other options for the device
parts = {'R':{'nodes':2,'value':1,'min':1,'max':1e6,'cost':0.1},
        'C':{'nodes':2,'value':1,'min':1e-10,'max':1e-3,'cost':0.1},
         #'L':{'nodes':2,'value':1,'min':1e-9,'max':1e-3},
         'X':{'nodes':3,'spice':'nc ne TL071','cost':0.5},
         }

def _fitness_function1(f,k,**kwargs):
    """k is the name of measurement. eq. v(n2)"""
    if k[0]=='v':
        #-100dB/decade
        return 43.43*log(f)-300 if f<=1000 else 0
    elif k[0]=='i':
        #Goal for current use is 0
        return 0

def _fitness_function2(f,k,**kwargs):
    return 0

def _constraint1(f,x,k,**kwargs):
    if k[0]=='v':
        if f>10000:
            return 0.5>x>-0.5
        elif f>1000:
            return 3>x>-3
        elif f<100:
            return x<-40
        else:
            return x<0
    return True

def _constraint2(f,x,k,**kwargs):
    if k[0]=='v':
        if f<1e-3:
            return abs(x)<2
        else:
            return abs(x)<1.1
    if k=='i(vin)':
        return abs(x)<0.01+0.1/kwargs['generation']
    if k[0]=='i':
        return abs(x)<0.02+0.1/kwargs['generation']
    return True

def _constraint3(f,x,k,**kwargs):
    if k[0]=='v':
        if f<10e-3:
            return abs(x)<0.2
        else:
            return abs(x)<0.1
    if k=='i(vin)':
        return abs(x)<0.01+0.1/kwargs['generation']
    if k[0]=='i':
        return abs(x)<0.02+0.1/kwargs['generation']
    return True

def _constraint4(f,x,k,**kwargs):
    if k[0]=='v':
        if f<1e-3:
            return abs(x)<0.1
        else:
            return abs(x)<0.03
    if k[0]=='i':
        return abs(x)<0.02+0.1/kwargs['generation']
    return True

population=2000#Too small population might not converge, or converges to local minimum, but is faster to simulate
max_parts=12
nodes=10
mutation_rate=0.75
crossover_rate=0.05
gradual_constraints = True
constraint_free_generations = 1
constraint_ramp = 20
plot_every_generation = True
fitness_function=[_fitness_function1,_fitness_function1,_fitness_function2,_fitness_function2,_fitness_function2]
fitness_weight=[{'vdb(n2)':lambda x,**kwargs:10 if 100<x else 3},{'vdb(n2)':lambda x,**kwargs:10 if 100<x else 3},{'i(vin)':0.1,'v(n2)':0,'i(vc)':0.05,'i(ve)':0.05},{'i(vin)':0.1,'v(n2)':5,'i(vc)':0.05,'i(ve)':0.05},{'i(vin)':0.5,'v(n2)':10,'i(vc)':0.1,'i(ve)':0.1}]
constraints=[_constraint1,_constraint1,_constraint2,_constraint3,_constraint4]
constraint_weight=[1000,1000,10000,1000,1000]
plot_yrange={'vdb(n2)':(-120,20),'i(vin)':(-0.2,0.2),'i(vc)':(-0.2,0.2),'i(ve)':(-0.2,0.2),'v(n2)':(-2,2)}
