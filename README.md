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
