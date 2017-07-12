import numpy as np
#import matplotlib.pyplot as plt
import xsmodels

import pytest


def pl_integral(ebins, params):
    print("params: " + str(params))
    idx = params[0]
    norm = params[1]
    flux = norm/(-idx + 1.) * (ebins[1:]**(-idx + 1.) - ebins[:-1]**(-idx + 1.))
    return flux

class TestXSModel(object):

    @classmethod
    def setup_class(cls):
        cls.nFlux = 4
        cls.e1 = 0.1
        cls.e2 = 10.0
    
        cls.ebins = np.logspace(np.log10(cls.e1), np.log10(cls.e2), cls.nFlux+1, base=10.0)
        cls.flux = np.empty(cls.nFlux)
        cls.flux_err = np.empty(cls.nFlux)
        cls.params = np.array([2.0, 1.0])
    
    def test_powerlaw(self):
        xsmodels.model("powerLaw", self.ebins, self.params, 0, self.flux, self.flux_err)
        assert np.all(self.flux > 0)         

    def test_powerlaw_results(self):
        true_flux = pl_integral(self.ebins, self.params)

        flux = np.empty(self.nFlux)
        flux_err = np.empty(self.nFlux)
        xsmodels.model("powerLaw", self.ebins, self.params, 0, flux, flux_err)
        print(true_flux)
        print(flux)
        assert np.allclose(flux, true_flux)

    def test_xsmodel_fails_with_unknown_model(self):
        flux = np.empty(self.nFlux)
        flux_err = np.empty(self.nFlux)

        with pytest.raises(ValueError):
            xsmodels.model("Hiya!", self.ebins, self.params, 0, flux, flux_err);

