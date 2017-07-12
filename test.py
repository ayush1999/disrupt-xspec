import numpy as np
import matplotlib.pyplot as plt
import xsmodels

def test_powerLaw():

    nFlux = 99
    e1 = 0.1
    e2 = 10.0

    ebins = np.logspace(np.log10(e1), np.log10(e2), nFlux+1, base=10.0)
    flux = np.empty(nFlux)
    flux_err = np.empty(nFlux)
    params = np.array([2.0, 1.0])

    print(ebins.shape)
    print(params.shape)
    print(flux.shape)
    print(flux_err.shape)
    print(ebins)
    print(params)
    print(flux)
    print(flux_err)

    xsmodels.powerLaw(ebins, params, 0, flux, flux_err)

    print(ebins)
    print(params)
    print(flux)
    print(flux_err)

    e = 0.5*(ebins[1:]+ebins[:-1])

    fig, ax = plt.subplots(1,1)
    ax.plot(e, flux, 'k+')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$E$ (keV)')
    ax.set_ylabel(r'Flux (?)')
    fig.savefig("flux_py.png")
    plt.close(fig)


def test_model():

    nFlux = 99
    e1 = 0.1
    e2 = 10.0

    ebins = np.logspace(np.log10(e1), np.log10(e2), nFlux+1, base=10.0)
    flux = np.empty(nFlux)
    flux_err = np.empty(nFlux)
    params = np.array([2.0, 1.0])

    xsmodels.model("Hiya!",ebins, params, 0, flux, flux_err);
    xsmodels.model("powerLaw",ebins, params, 0, flux, flux_err);

    e = 0.5*(ebins[1:]+ebins[:-1])

    fig, ax = plt.subplots(1,1)
    ax.plot(e, flux, 'k+')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$E$ (keV)')
    ax.set_ylabel(r'Flux (?)')
    fig.savefig("flux_model_py.png")
    plt.close(fig)

if __name__ == "__main__":

    test_powerLaw()
    test_model()
