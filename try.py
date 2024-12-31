import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def read_mumax3_table(filename):
    """Puts the mumax3 output table in a pandas dataframe"""

    from pandas import read_table
    
    table = read_table(filename)
    table.columns = ' '.join(table.columns).split()[1::2]
    
    return table

def read_mumax3_ovffiles(outputdir):
    """Load all ovffiles in outputdir into a dictionary of numpy arrays 
    with the ovffilename (without extension) as key"""
    
    from subprocess import run, PIPE, STDOUT
    from glob import glob
    from os import path
    from numpy import load

    # convert all ovf files in the output directory to numpy files
    p = run(["mumax3-convert","-numpy",outputdir+"/*.ovf"], stdout=PIPE, stderr=STDOUT)
    if p.returncode != 0:
        print(p.stdout.decode('UTF-8'))

    # read the numpy files (the converted ovf files)
    fields = {}
    for npyfile in glob(outputdir+"/*.npy"):
        key = path.splitext(path.basename(npyfile))[0]
        fields[key] = load(npyfile)
    
    return fields

def run_mumax3(name, verbose=False):
    """ Executes a mumax3 script and convert ovf files to numpy files
    
    Parameters
    ----------
      script:  string containing the mumax3 input script
      name:    name of the simulation (this will be the name of the script and output dir)
      verbose: print stdout of mumax3 when it is finished
    """
    
    from subprocess import run, PIPE, STDOUT
    from os import path

    #scriptfile = name + ".txt" 
    outputdir  = name + ".out"

    # write the input script in scriptfile
    #with open(scriptfile, 'w' ) as f:
    #    f.write(script)
    
    # call mumax3 to execute this script
    #p = run(["mumax3","-f",scriptfile], stdout=PIPE, stderr=STDOUT)
    #if verbose or p.returncode != 0:
    #    print(p.stdout.decode('UTF-8'))
        
    if path.exists(outputdir + "/table.txt"):
        table = read_mumax3_table(outputdir + "/table.txt")
    else:
        table = None
        
    fields = read_mumax3_ovffiles(outputdir)
    
    return table, fields

table, fields = run_mumax3( name="SHNO10t5", verbose=False )

print(table)

plt.figure()

nanosecond = 1e-9
plt.plot( table["t"]/nanosecond, table["mx"])
plt.plot( table["t"]/nanosecond, table["my"])
plt.plot( table["t"]/nanosecond, table["mz"])

plt.xlabel("Time (ns)")
plt.ylabel("Magnetization")

plt.show()
#print(fields.keys())

m = fields["m003000"]

print("type  =", type(m))
print("shape =", m.shape)

def show_abs_my(m):
    my_abs = np.abs( m[1,0,:,:] )
    plt.figure()
    plt.imshow(my_abs, vmin=0, vmax=1, cmap="afmhot")
    plt.show()

show_abs_my(fields["m003001"])

fmax = 50e9        # maximum frequency (in Hz) of the sinc pulse
T    = 1e-9        # simulation time (longer -> better frequency resolution)
dt   = 1/(2*fmax)  # the sample time (Nyquist theorem taken into account)

# FAST FOURIER TRANSFORM
dm     = table["my"] - table["my"][0]   # average magnetization deviaton
spectr = np.abs(np.fft.fft(dm))         # the absolute value of the FFT of dm
freq   = np.linspace(0, 1/dt, len(dm))  # the frequencies for this FFT

# PLOT THE SPECTRUM
plt.plot(freq/1e9, spectr)
plt.xlim(0,fmax/1e9)
plt.ylabel("Spectrum (a.u.)")
plt.xlabel("Frequency (GHz)")
plt.show()

# MODE IDENTIFICATION
mode1_idx = 32   # determined visually
mode2_idx = 64  # determined visually
mode1_freq = freq[mode1_idx]
mode2_freq = freq[mode2_idx]
print("Mode 1 frequency: %.2f GHz"%(mode1_freq/1e9))
print("Mode 2 frequency: %.2f GHz"%(mode2_freq/1e9))

# PLOT THE SPECTRUM
plt.plot(freq/1e9, spectr)
plt.axvline(mode1_freq/1e9, lw=1, ls='--', c='gray')
plt.axvline(mode2_freq/1e9, lw=1, ls='--', c='gray')
plt.xlim(0,fmax/1e9)
plt.ylabel("Spectrum (a.u.)")
plt.xlabel("Frequency (GHz)")
plt.show()

# Stack all snapshots (4D arrays) of the magnetization on top of each other
# The results in a single 5D array (first index is the snapshot index)
m = np.stack([fields[key] for key in sorted(fields.keys())])

# Select the z component and the (only) layer z=0
my = m[:,1,0,:,:]

# Apply the FFT for every cell
my_fft = np.fft.fft(my, axis=0)

# Select the the two modes
mode1 = my_fft[mode1_idx]
mode2 = my_fft[mode2_idx]

# Plot the result
plt.figure(figsize=(10,4))
plt.subplot(1,3,1)
plt.title("$m_y$")
plt.imshow(my[0])
plt.subplot(1,3,2)
plt.title("Mode 1")
plt.imshow(np.abs(mode1)**2)
plt.subplot(1,3,3)
plt.title("Mode 2")
plt.imshow(np.abs(mode2)**2)
plt.show()