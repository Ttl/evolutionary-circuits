from math import sqrt
title = 'MOSFET-Sqrt2012-08-22'

models="""
.SUBCKT BS170 3 4 5
*             D G S
M1 3 2 5 5 N3306M
RG 4 2 270
RL 3 5 1.2E8
C1 2 5 28E-12
C2 3 2 3E-12
D1 5 3 N3306D
*
.MODEL N3306M NMOS VTO=1.824 RS=1.572 RD=1.436 IS=1E-15 KP=.1233
+CBD=35E-12 PB=1
.MODEL N3306D D IS=5E-12 RS=.768
.ENDS BS170
.SUBCKT BS250 1 2 3
* 1=drain  2=gate  3=source
Cgs  2 3 20.1E-12
Cgd1 2 4 57.1E-12
Cgd2 1 4 5E-12
M1 1 2 3 3 MOST1
M2 4 2 1 3 MOST2
D1 1 3 Dbody
.MODEL MOST1 PMOS(LEVEL=3 VTO=-2.3 W=7.6m L=2u KP=10.33u RD=4.014 RS=20m)
.MODEL MOST2 PMOS(VTO=2.43 W=7.6m L=2u KP=10.33u RS=20m)
.MODEL Dbody D(CJO=53.22E-12 VJ=0.5392 M=0.3583 IS=75.32E-15 N=1.016 RS=1.245
+              TT=86.56n BV=45 IBV=10u)
.ENDS BS250
"""

common="""
vc nc 0 10
ve ne 0 -10
Rc nc n0 1
Re ne n3 1
rload n2 0 100k
"""

spice_commands=[
"""
.control
dc Vin 0 5 0.02
print v(n2)
print i(Vin)
print i(ve)
print i(vc)
.endc
Vin n1 0
""",
"""
.control
tran 1u 2m
print v(n2)
print i(vin)
print i(ve)
print i(vc)
.endc
Vin n1 0 PULSE(1 5 10u 5u 5u 1 1)
""",
"""
.control
tran 1u 2m
print v(n2)
print i(vin)
print i(vc)
print i(ve)
.endc
Vin n1 0 PULSE(4 0 10u 5u 5u 1 1)
"""
]

#Dictionary of the availabe parts
parts = {'R':{'nodes':2,'value':1,'min':1,'max':6,'cost':0.1},
        #'C':{'nodes':2,'value':1,'min':-13,'max':-6,'cost':0.1},
         #'L':{'nodes':2,'value':1,'min':-9,'max':-3},
         #'D1':{'nodes':2,'spice':'1N4148'},
         #'Q1':{'nodes':3,'spice':'2N3904'},
         #'Q2':{'nodes':3,'spice':'2N3906'},
         'X1':{'nodes':3,'spice':'BS170','cost':0.2},
         'X2':{'nodes':3,'spice':'BS250','cost':0.2}
         }

def _goalinv(f,k,**kwargs):
    """k is the name of measurement. eq. v(n2)"""
    #This is the DC transfer curve goal function
    if k=='v(n2)':
        #kwargs['extra'][0] is the transition voltage
        return sqrt(f)*kwargs['extra'][0]
    elif k[0]=='i':
        #Goal for current use is 0
        return 0

def _transient_goal_inv(f,k,**kwargs):
    #Goal for first transient simulation
    u = 10**(-6)
    if k[0]=='v':
        if f<=10*u:
            return 1*kwargs['extra'][0]
        else:
            return 2.23606797749979*kwargs['extra'][0]
    return 0#For current

def _transient_goal_inv2(f,k,**kwargs):
    #Second transient simulation
    u = 10**(-6)
    if k[0]=='v':
        if f<=10*u:
            return 2*kwargs['extra'][0]
        else:
            return 0
    return 0

def _constraint1(f,x,k,**kwargs):
    if k == 'v(n2)':
        if f<0.1:
            return abs(x-sqrt(f)*kwargs['extra'][0])<0.01
        if f<1:
            return abs(x-sqrt(f)*kwargs['extra'][0])<0.1
    return True

population=1000
max_parts=28#Maximum number of parts
nodes=22
elitism=1#Best circuit is copied straight to next generation
mutation_rate=0.7
crossover_rate=0.02
fitness_function=[_goalinv,_transient_goal_inv,_transient_goal_inv2]
fitness_weight=[{'v(n2)':lambda x,**kwargs:1000/kwargs['extra'][0]+10*abs(kwargs['extra'][0]-1),'i(vc)':100,'i(ve)':100,'i(vin)':100},{'v(n2)':2,'i(vc)':30,'i(ve)':30,'i(vin)':30},{'v(n2)':2,'i(vc)':30,'i(ve)':30,'i(vin)':30}]#Current use is weighted very heavily
extra_value=[(0.5,2.0)]
constraints=[_constraint1,None,None]
constraint_weight=[10000,0,0]
plot_titles=[{'v(n2)':"DC sweep",'i(vc)':"Current from power supply",'i(vin)':'Current from logic input'},{'v(n2)':'Step response'},{'v(n2)':'Step response'}]
plot_yrange={'v(n2)':(0,5),'i(vin)':(-0.2,0.01),'i(vc)':(-0.2,0.01)}
