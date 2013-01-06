import random
from math import log10

def value_dist(val):
    if len(val)==3:
        return val[3](*val[:2])
    else:
        if val[0]>0:
            return log_dist(*val)
        else:
            return random.uniform(*val)

def argmin(x):
    if len(x) < 2:
        return x[0]
    bestv,idx = x[0],0
    for e,i in enumerate(x[1:],1):
        if i<bestv:
            bestv = i
            idx = e
    return idx

def argmax(x):
    if len(x) < 2:
        return x[0]
    bestv,idx = x[0],0
    for e,i in enumerate(x[1:],1):
        if i>bestv:
            bestv = i
            idx = e
    return idx

def normalize_list(lst):
    """Return normalized list that sums to 1"""
    s = sum(lst)
    return [i/float(s) for i in lst]

def multipliers(x):
    """Convert values with si multipliers to numbers"""
    try:
        return float(x)
    except:
        pass
    try:
        a = x[-1]
        y = float(x[:-1])
        endings = {'G':9,'Meg':6,'k':3,'m':-3,'u':-6,'n':-9,'p':-12,'s':0}
        return y*(10**endings[a])
    except:
        raise ValueError("I don't know what {} means".format(x))

def log_dist(a,b):
    """Generates exponentially distributed random numbers.
    Gives better results for resistor, capacitor and inductor values
    than the uniform distribution."""
    if a <= 0 or a>b:
        raise ValueError("Value out of range. Valid range is (0,infinity).")
    return 10**(random.uniform(log10(a),log10(b)))

def same(x):
    #True if all elements are same
    return reduce(lambda x,y:x==y,x)

def lst_random(lst, probs):
    """Return element[i] with probability probs[i]."""
    s = sum(probs)
    r = random.uniform(0,s)
    t = 0
    for i in xrange(len(lst)):
        t += probs[i]
        if r <= t:
            return lst[i]
    return lst[-1]#Because of rounding errors or something?
