import sys
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":

    fname = sys.argv[1]

    f = open(fname, "r")
    line1 = f.readline()
    line2 = f.readline()
    line3 = f.readline()
    f.close()

    Ebin = np.array([float(x) for x in line1.split()[1:]])
    F = np.array([float(x) for x in line2.split()[1:]])
    Ferr = np.array([float(x) for x in line3.split()[1:]])

    E = 0.5*(Ebin[1:] + Ebin[:-1])
    FE = F/E

    fig, ax = plt.subplots(1,1)
    ax.plot(E, F, 'k+')
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$E$ (keV)")
    ax.set_ylabel(r"$F_E$")
    fig.savefig("flux.png")
