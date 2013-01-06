title = 'Inverter'

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

#This is for constructions common for every simulation
common="""
rin na in 10
vc vc 0 5
rload out 0 10k
cload out 0 100p
"""

#List of simulations
#Put SPICE simulation options between.control and .endc lines
spice_commands=[
"""
.control
dc Vin 0 5 0.05
print v(out)
print i(Vin)
print i(vc)
.endc
Vin na 0 0
""",
"""
.control
tran 2n 800n
print v(out)
print i(vin)
print i(vc)
.endc
Vin na 0 PULSE(0 5 10n 1n 1n 1 1)
""",
"""
.control
tran 2n 800n
print v(out)
print i(vin)
print i(vc)
.endc
Vin na 0 PULSE(5 0 10n 1n 1n 1 1)
"""
]

inputs = ['in']
outputs = ['out']
#Power supply node
special_nodes = ['vc']

#Dictionary of the availabe parts
parts = {'R':{'nodes':2,'value':(0.1,1e5)},#Resistors
         #'C':{'nodes':2,'value':1,'min':1e-13,'max':1e-9},#Capacitors
         #'L':{'nodes':2,'value':1,'min':1e-9,'max':1e-3},#No inductors allowed
         'Q':{'nodes':3,'model':('2N3904','2N3906')},#NPN transistors
         }

#Names starting with underscore are not loaded
def _goalinv(f,k,**kwargs):
    """k is the name of measurement. eq. v(out)"""
    #This is the DC transfer curve goal function
    if k=='v(out)':
        #kwargs['extra'][0] is the transition voltage
        return 5 if f<=kwargs['extra'][0] else 0
    elif k[0]=='i':
        #Goal for current use is 0
        return 0

def _transient_goal_inv(f,k,**kwargs):
    #Goal for first transient simulation
    n = 10**(-9)
    if k[0]=='v':
        if f<=10*n:
            return 5
        elif 20*n<f:
            return 0
    return 0#For current

def _transient_goal_inv2(f,k,**kwargs):
    #Second transient simulation
    n = 10**(-9)
    if k[0]=='v':
        if f<=10*n:
            return 0
        elif 20*n<f:
            return 5
    return 0

def _constraint0(f,x,k,**kwargs):
    #Constraint for static current consumption
    if k[0]=='v':
        if f<=kwargs['extra'][0]-0.2:
            return x>kwargs['extra'][0]+0.2
        elif f>=kwargs['extra'][0]+0.2:
            return x<kwargs['extra'][0]-0.2
    if k[0]=='i':
        #Limits current taken and going into the logic input to 2mA
        #Currents taken from the supply are negative and currents going into to
        #the supply are positive
        return abs(x)<5e-3+0.1/kwargs['generation']**0.5
    return True

def _constraint1(f,x,k,**kwargs):
    """f is input, x is output, k is measurement, extra is extra value of chromosome"""
    #Constraint for the first transient simulation
    if k[0]=='v' and f<9e-9:
        #Output should be 0.2V above the transition voltage at t=0
        return x>kwargs['extra'][0]+0.2
    if k[0]=='v' and f>350e-9:
        #And below it after the transition on the input
        return x<kwargs['extra'][0]-0.2
    if k[0]=='i':
        #Goal for current use
        return abs(x)<10e-3+0.1/kwargs['generation']**0.5
    return True

def _constraint2(f,x,k,**kwargs):
    """f is input, x is output, k is measurement, extra is extra value of chromosome"""
    #Same as last one, but with other way around
    if k[0]=='v' and f<9e-9:
        return x<kwargs['extra'][0]-0.2
    if k[0]=='v' and f>350e-9:
        return x>kwargs['extra'][0]+0.2
    if k[0]=='i':
        return abs(x)<10e-3+0.1/kwargs['generation']**0.5
    return True

population=500#Too small population might not converge, but is faster to simulate
max_parts=10#Maximum number of parts

mutation_rate=0.70
crossover_rate=0.10

#Because constraint functions change every generation score might
#increase even when better circuit is found
plot_every_generation = True
fitness_function=[_goalinv,_transient_goal_inv,_transient_goal_inv2]
constraints=[_constraint0,_constraint1,_constraint2]
constraint_weight=[1000,1000,1000]
constraint_free_generations = 1
gradual_constraints = True
constraint_ramp = 30
fitness_weight=[{'v(out)':lambda f,**kwargs: 15 if (f<0.5 or f>4.5) else 0.1,'i(vc)':lambda f,**kwargs:kwargs['generation']*100,'i(vin)':lambda f,**kwargs:kwargs['generation']*100},{'v(out)':2,'i(vc)':lambda f,**kwargs:kwargs['generation']*100,'i(vin)':1000},{'v(out)':2,'i(vc)':lambda f,**kwargs:kwargs['generation']*100,'i(vin)':1000}]

extra_value=[(0.5,4.5)]#This is used as transition value

plot_titles=[{'v(out)':"DC sweep",'i(vc)':"Current from power supply",'i(vin)':'Current from logic input'},{'v(out)':'Step response'},{'v(out)':'Step response'}]
plot_yrange={'v(out)':(-0.5,5.5),'i(vin)':(-0.05,0.01),'i(vc)':(-0.05,0.01)}
