import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from os.path import join as path_join
from time import strftime

def save_plot(v,k,goal_val,sim_type,generation,score,output_path,plot_title=None,yrange=None,log_plot=False,constraints=None,name=''):
    #For every measurement in results
    plt.figure()
    freq = v[0]
    gain = v[1]
    if log_plot:#Logarithmic plot
        plt.semilogx(freq,gain,'g',basex=10)
        plt.semilogx(freq,goal_val,'b',basex=10)
        #if self.plot_weight:
        #    plt.semilogx(freq,weight_val,'r--',basex=10)
        if constraints!=None:
            plt.plot(*zip(*constraints), marker='.', color='r', ls='')
    else:
        plt.plot(freq,gain,'g')
        plt.plot(freq,goal_val,'b')
        #if self.plot_weight:
        #    plt.plot(freq,weight_val,'r--')
        if constraints!=None:
            plt.plot(*zip(*constraints), marker='.', color='r', ls='')

    # update axis ranges
    ax = []
    ax[0:4] = plt.axis()
    # check if we were given a frequency range for the plot
    if yrange!=None:
        plt.axis([min(freq),max(freq),yrange[0],yrange[1]])
    else:
        plt.axis([min(freq),max(freq),min(-0.5,-0.5+min(goal_val)),max(1.5,0.5+max(goal_val))])

    if sim_type=='dc':
        plt.xlabel("Input (V)")
    if sim_type=='ac':
        plt.xlabel("Input (Hz)")
    if sim_type=='tran':
        plt.xlabel("Time (s)")

    if plot_title!=None:
        plt.title(plot_title)
    else:
        plt.title(k)

    plt.annotate('Generation '+str(generation),xy=(0.05,0.95),xycoords='figure fraction')
    if score!=None:
        plt.annotate('Score '+'{0:.2f}'.format(score),xy=(0.75,0.95),xycoords='figure fraction')
    plt.grid(True)
    # turn on the minor gridlines to give that awesome log-scaled look
    plt.grid(True,which='minor')
    if len(k)>=3 and k[1:3] == 'db':
        plt.ylabel("Output (dB)")
    elif k[0]=='v':
        plt.ylabel("Output (V)")
    elif k[0]=='i':
        plt.ylabel("Output (A)")

    plt.savefig(path_join(output_path,strftime("%Y-%m-%d %H:%M:%S")+'-'+k+'-'+str(name)+'.png'))

data = input()
save_plot(*data)
