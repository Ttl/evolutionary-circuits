evolutionary-circuits
=====================

Automatically generates analog circuits using evolutionary algorithms.

Requires ngspice. Also probably only works on a linux OS.

Interesting stuff happens on cgp.py. circuits.py is used to communicate with
ngspice, plot.py is not used by program but it can be used to plot outputs of
ngspice. getch.py gets a single character from output.

Work is currently very much in progress and documentation is missing. This is
currently very hard to use and requires a knowledge of how to use SPICE.

Simulation can be stopped with a keyboard interrupt and it can be also resumed
without losing progress.

#Installing

##ngspice

Download [ngpspice](http://ngspice.sourceforge.net/) either git version or the
current stable, whichever you want.

Compile and install ngspice with following commands:

     $ mkdir release
     $ cd release
     $ ../configure --disable-debug
     $ make 2>&1 | tee make.log
     $ sudo make install

#Usage

Currently program is used by changing simulation parameters in cgp.py file, but
hopefully current simulation options are moved to a different file soon.

First you need to decide what components do you want to allow and add them to
the "parts" dictionary. Dictionary key is the name of component in SPICE, value
should be an another dictionary with contents:
    
    'nodes': Number of leads in the component
    'value': True/False, component has a value(Example: True for resistors and
    capacitors, their values are resistance and capacitance.
    False for example transistors)
    'min': Exponent of the components minimum value. Minimum value will be
    10^(min)
    'max': Exponent of the components maximum value.
    'spice': Extra options for spice. This is appended to the end of components
    description in spice. This is used for example transistors model.

Then you need SPICE simulation commands and part models. They should be strings
in a list. Every item in list is a new spice simulation. Currently ouput node is
hard coded to be 'n2' and the ground node is always 0. Add "print v(n2)" or "print i(n2)" in the spice commands to
get an output to the program. You can print more than one measurement from one
spice simulation.

Example: spice DC simulation that sweep Vin power supply from 0V to
10V with a step of 0.1V and also has a one fixed load resistance and 100ns long transient simulation to a step response. Both simulations include QMOD and QMOD2 transistor models:

    options="""
    .control
    dc Vin 0 10 0.1
    print v(n2)
    .endc
    .MODEL QMOD NPN(BF=100 CJC=20pf CJE=20pf IS=1E-16)
    Vin n1 0
    rload n2 0 100k
    """,
    """
    .control
    tran 0.5n 200n
    print v(n2)
    .endc
    .MODEL QMOD NPN(BF=100 CJC=20pf CJE=20pf IS=1E-16)
    Vin n1 0 PULSE(0 10 10n 1n 1n 1 1)
    rload n2 0 100k
    """

Other settings you should specify are a list of fitness functions. These are
functions that evolved circuits are the goals of the simulation.

For example a 1V/1A constant fitness function:

    def goal(v,k):
        """v is input, either voltage, frequency, time... depends on spice
        simulation. k is the name of node being measured"""
        return 1

Another fitness function if spice simualtion has more than one measurement. This
example has a current and voltage measurements:

    def goal(v,k):
        if k[0]=='v':   #Voltage
            return 1    #1V
        elif k[0]=='i': #Current
            return 1e-3 #1mA

Discontinous fitness function, that is 0 before time is 100ns and 5 after it:

    def goal(t,k):
        #t is now time
        return 0 if t<=100e-9 else 5

Fitness weights are either constants that scores are multiplied with, or
functions that invidual values of spice simulation are weighted with if you want
to for example weight some part of the results higher than other parts. They
should be in list of dictionaries where number of elements in list is same as
number of spice simulations. nth element is used with nth spice simulation. Keys
of the dictionaries are names of measurements(eg. 'v(n2)') and values are
weights.

For example a constants weights with 4 simulations with second one having two
measurements: 

    fitness_weight=[{'v(n2)':1},{'v(n2)':1,'i(vin)':50},{'v(n2)':1},{'v(n2)':0.05}]


Other options that should be specified:
    
    pool_size: Number of circuits in one generation
    nodes: Number of available nodes in one circuit
    max_parts: Maximum number of parts in one circuit
    elitism: Number of circuits that are copied to next generation with 100%
    probability.
    mutation_rate: Probability of mutations(In range 0.0-1.0)
    crossover_rate: Probability of crossovers
    log: Log file
    plot_titles: Titles of plots, list of dictionaries. Same format as
    fitness_weight
    plot_yrange: Range of y-axis in the plots, dictionary with measurements as
    keys and tuple of (min,max) as values.


See the bottom of "cgp.py" for examples.
