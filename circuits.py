import subprocess
import threading

def parse_output(output):
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

        if len(temp)>2 and current!=():
            try:
                float(temp[1]),float(temp[2])
            except:
                continue
            index+=1
            for i in xrange(2):
                value[current][0].append(float(temp[1]))
                value[current][1].append(float(temp[2]))
    return value

class spice_thread(threading.Thread):
    def __init__(self, spice_in):
        threading.Thread.__init__(self)
        self.spice_in = spice_in
        self.result = None
        if self.spice_in!=None:
            self.spice = subprocess.Popen(['ngspice','-n'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    def run(self):
        if self.spice_in!=None:
            output = self.spice.communicate(self.spice_in)
            self.result = (output[1],parse_output(output[0]))
