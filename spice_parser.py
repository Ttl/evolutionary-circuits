from cgp import Circuit_gene

def parse_circuit(circuit,parts):
    devices = []
    #Devices starting with same substring are sorted longest
    #first to check longest possible device names first
    sorted_dev = sorted(parts.keys(),reverse=True)
    for n,line in enumerate(circuit.splitlines()):
        if not line:
            #Ignores empty lines
            continue

        #Current device fields
        d_spice = []
        #Try all the devices
        for dev in sorted_dev:
            if line.startswith(dev):
                #Found matching device from parts list
                line = line.split()
                d_nodes = line[1:parts[dev]['nodes']+1]
                d_spice = line[parts[dev]['nodes']+1:]
                devices.append(Circuit_gene(dev,d_nodes,0,d_spice))
                break

        else:
            #Device not found
            print "Couldn't find device in line {}:{}\nIgnoring this line".format(n,line)
    return devices
