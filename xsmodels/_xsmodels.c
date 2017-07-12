#include <stdio.h>
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_11_API_VERSION
#include <numpy/arrayobject.h>
#include <xsFortran.h>
#include <funcWrappers.h>

static char xsmodels_docstring[] = 
    "This module provides an interface to Xspec models.";

static char powerLaw_docstring[] = 
    "Calculate a spectrum with the Xspec powerLaw model.";

static char model_docstring[] = 
    "Calculate a spectrum with the a specified Xspec model.";

static PyObject *xsmodels_powerLaw(PyObject *self, PyObject *args);
static PyObject *xsmodels_model(PyObject *self, PyObject *args);

static PyMethodDef xsmodels_methods[] = {
    {"powerLaw", xsmodels_powerLaw, METH_VARARGS, powerLaw_docstring},
    {"model", xsmodels_model, METH_VARARGS, model_docstring},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef xsmodels = {
    PyModuleDef_HEAD_INIT,
    "_xsmodels",  /* Module Name*/
    xsmodels_docstring, /* Module docstring */
    -1, 
    xsmodels_methods
};

int xspec_model(const char model[], double *energy, int nFlux, double *params, 
                    int spectrumNumber, double *flux, double *fluxError,
                    const char initStr[]);


PyMODINIT_FUNC PyInit__xsmodels(void)
{
    //XSpec interface initialization
    FNINIT();

    //Load numpy stuff!
    import_array();

    return PyModule_Create(&xsmodels);
}

static PyObject *xsmodels_powerLaw(PyObject *self, PyObject *args)
{
    int spectrumNumber;
    PyObject *energy_obj = NULL;
    PyObject *params_obj = NULL;
    PyObject *flux_obj = NULL;
    PyObject *fluxError_obj = NULL;
    const char *initStr = NULL;

    //Parse Arguments
    if(!PyArg_ParseTuple(args, "OOiOO|s", &energy_obj, &params_obj, 
                        &spectrumNumber, &flux_obj, &fluxError_obj, &initStr))
    {
        PyErr_SetString(PyExc_RuntimeError, "Could not parse arguments.");
        return NULL;
    }
    //Grab numpy arrays
    PyArrayObject *energy_arr;
    PyArrayObject *params_arr;
    PyArrayObject *flux_arr;
    PyArrayObject *fluxError_arr;

    //Grab arrays and test if read properly
    energy_arr    = (PyArrayObject *) PyArray_FROM_OTF(energy_obj,
                                            NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    params_arr    = (PyArrayObject *) PyArray_FROM_OTF(params_obj,
                                            NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    flux_arr      = (PyArrayObject *) PyArray_FROM_OTF(flux_obj, 
                                            NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY);
    fluxError_arr = (PyArrayObject *) PyArray_FROM_OTF(fluxError_obj, 
                                            NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY);

    if(energy_arr == NULL || params_arr == NULL || flux_arr == NULL
            || fluxError_arr == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "Could not read input arrays.");
        Py_XDECREF(energy_arr);
        Py_XDECREF(params_arr);
        Py_XDECREF(flux_arr);
        Py_XDECREF(fluxError_arr);
        return NULL;
    }

    int energy_ndim = (int) PyArray_NDIM(energy_arr);
    int params_ndim = (int) PyArray_NDIM(params_arr);
    int flux_ndim = (int) PyArray_NDIM(flux_arr);
    int fluxError_ndim = (int) PyArray_NDIM(fluxError_arr);

    if(energy_ndim != 1 || params_ndim != 1 || flux_ndim != 1 
            || fluxError_ndim != 1)
    {
        PyErr_SetString(PyExc_RuntimeError, "Arrays must be 1-D.");
        Py_DECREF(energy_arr);
        Py_DECREF(params_arr);
        Py_DECREF(flux_arr);
        Py_DECREF(fluxError_arr);
        return NULL;
    }

    int nFlux = (int)PyArray_DIM(flux_arr, 0);
    int fluxError_dim = (int)PyArray_DIM(fluxError_arr, 0);
    int energy_dim = (int)PyArray_DIM(energy_arr, 0);
    if(fluxError_dim != nFlux || energy_dim != nFlux+1)
    {
        PyErr_SetString(PyExc_RuntimeError, "energy must be 1 longer than flux and fluxError.");
        Py_DECREF(energy_arr);
        Py_DECREF(params_arr);
        Py_DECREF(flux_arr);
        Py_DECREF(fluxError_arr);
        return NULL;
    }

    double *energy = (double *)PyArray_DATA(energy_arr);
    double *params = (double *)PyArray_DATA(params_arr);
    double *flux = (double *)PyArray_DATA(flux_arr);
    double *fluxError = (double *)PyArray_DATA(fluxError_arr);

    C_powerLaw(energy, nFlux, params, spectrumNumber, flux, fluxError, 
                initStr);

    Py_DECREF(energy_arr);
    Py_DECREF(params_arr);
    Py_DECREF(flux_arr);
    Py_DECREF(fluxError_arr);

    Py_RETURN_NONE;
}

static PyObject *xsmodels_model(PyObject *self, PyObject *args)
{
    int spectrumNumber;
    PyObject *energy_obj = NULL;
    PyObject *params_obj = NULL;
    PyObject *flux_obj = NULL;
    PyObject *fluxError_obj = NULL;
    const char *modelStr = NULL;
    const char *initStr = NULL;

    //Parse Arguments
    if(!PyArg_ParseTuple(args, "sOOiOO|s", &modelStr, &energy_obj, &params_obj, 
                        &spectrumNumber, &flux_obj, &fluxError_obj, &initStr))
    {
        PyErr_SetString(PyExc_RuntimeError, "Could not parse arguments.");
        return NULL;
    }
    //Grab numpy arrays
    PyArrayObject *energy_arr;
    PyArrayObject *params_arr;
    PyArrayObject *flux_arr;
    PyArrayObject *fluxError_arr;

    //Grab arrays and test if read properly
    energy_arr    = (PyArrayObject *) PyArray_FROM_OTF(energy_obj,
                                            NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    params_arr    = (PyArrayObject *) PyArray_FROM_OTF(params_obj,
                                            NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    flux_arr      = (PyArrayObject *) PyArray_FROM_OTF(flux_obj, 
                                            NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY);
    fluxError_arr = (PyArrayObject *) PyArray_FROM_OTF(fluxError_obj, 
                                            NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY);

    if(energy_arr == NULL || params_arr == NULL || flux_arr == NULL
            || fluxError_arr == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "Could not read input arrays.");
        Py_XDECREF(energy_arr);
        Py_XDECREF(params_arr);
        Py_XDECREF(flux_arr);
        Py_XDECREF(fluxError_arr);
        return NULL;
    }

    int energy_ndim = (int) PyArray_NDIM(energy_arr);
    int params_ndim = (int) PyArray_NDIM(params_arr);
    int flux_ndim = (int) PyArray_NDIM(flux_arr);
    int fluxError_ndim = (int) PyArray_NDIM(fluxError_arr);

    if(energy_ndim != 1 || params_ndim != 1 || flux_ndim != 1 
            || fluxError_ndim != 1)
    {
        PyErr_SetString(PyExc_RuntimeError, "Arrays must be 1-D.");
        Py_DECREF(energy_arr);
        Py_DECREF(params_arr);
        Py_DECREF(flux_arr);
        Py_DECREF(fluxError_arr);
        return NULL;
    }

    int nFlux = (int)PyArray_DIM(flux_arr, 0);
    int fluxError_dim = (int)PyArray_DIM(fluxError_arr, 0);
    int energy_dim = (int)PyArray_DIM(energy_arr, 0);
    if(fluxError_dim != nFlux || energy_dim != nFlux+1)
    {
        PyErr_SetString(PyExc_RuntimeError, "energy must be 1 longer than flux and fluxError.");
        Py_DECREF(energy_arr);
        Py_DECREF(params_arr);
        Py_DECREF(flux_arr);
        Py_DECREF(fluxError_arr);
        return NULL;
    }

    double *energy = (double *)PyArray_DATA(energy_arr);
    double *params = (double *)PyArray_DATA(params_arr);
    double *flux = (double *)PyArray_DATA(flux_arr);
    double *fluxError = (double *)PyArray_DATA(fluxError_arr);

    int err = xspec_model(modelStr, energy, nFlux, params, spectrumNumber, flux,
                    fluxError, initStr);
    if(err) {
         PyErr_SetString(PyExc_ValueError, "Model not found!");
         return (PyObject *) NULL;
         }

    Py_DECREF(energy_arr);
    Py_DECREF(params_arr);
    Py_DECREF(flux_arr);
    Py_DECREF(fluxError_arr);

    Py_RETURN_NONE;
}

int xspec_model(const char model[], double *energy, int nFlux, double *params, 
                    int specNum, double *flux, double *fluxErr,
                    const char initStr[])
{
    int err = 0;

    if(!strcmp(model, "agauss"))
        C_agauss(energy, nFlux, params, specNum, flux, fluxErr, initStr);
    else if(!strcmp(model, "apec"))
        C_apec(energy, nFlux, params, specNum, flux, fluxErr, initStr);
    else if(!strcmp(model, "bapec"))
        C_bapec(energy, nFlux, params, specNum, flux, fluxErr, initStr);
    else if(!strcmp(model, "btapec"))
        C_btapec(energy, nFlux, params, specNum, flux, fluxErr, initStr);
    else if(!strcmp(model, "xsbexrav"))
        C_xsbexrav(energy, nFlux, params, specNum, flux, fluxErr, initStr);
    else if(!strcmp(model, "xsbexriv"))
        C_xsbexriv(energy, nFlux, params, specNum, flux, fluxErr, initStr);
    else if(!strcmp(model, "brokenPowerLaw"))
        C_brokenPowerLaw(energy, nFlux, params, specNum, flux, fluxErr, initStr);
    else if(!strcmp(model, "broken2PowerLaw"))
        C_broken2PowerLaw(energy, nFlux, params, specNum, flux, fluxErr, initStr);
    else if(!strcmp(model, "bvapec"))
        C_bvapec(energy, nFlux, params, specNum, flux, fluxErr, initStr);
    else if(!strcmp(model, "bvtapec"))
        C_bvtapec(energy, nFlux, params, specNum, flux, fluxErr, initStr);
    else if(!strcmp(model, "bvvapec"))
        C_bvvapec(energy, nFlux, params, specNum, flux, fluxErr, initStr);
    else if(!strcmp(model, "bvvtapec"))
        C_bvvtapec(energy, nFlux, params, specNum, flux, fluxErr, initStr);
    else if(!strcmp(model, "carbatm"))
        C_carbatm(energy, nFlux, params, specNum, flux, fluxErr, initStr);
    else if(!strcmp(model, "cemVMekal"))
        C_cemVMekal(energy, nFlux, params, specNum, flux, fluxErr, initStr);
    else if(!strcmp(model, "xscflw"))
        C_xscflw(energy, nFlux, params, specNum, flux, fluxErr, initStr);
    else if(!strcmp(model, "xscompps"))
        C_xscompps(energy, nFlux, params, specNum, flux, fluxErr, initStr);
    else if(!strcmp(model, "xscompth"))
        C_xscompth(energy, nFlux, params, specNum, flux, fluxErr, initStr);
    else if(!strcmp(model, "cplinear"))
        C_cplinear(energy, nFlux, params, specNum, flux, fluxErr, initStr);
    else if(!strcmp(model, "cutoffPowerLaw"))
        C_cutoffPowerLaw(energy, nFlux, params, specNum, flux, fluxErr, initStr);
    else if(!strcmp(model, "xseqpair"))
        C_xseqpair(energy, nFlux, params, specNum, flux, fluxErr, initStr);
    else if(!strcmp(model, "xseqth"))
        C_xseqth(energy, nFlux, params, specNum, flux, fluxErr, initStr);
    else if(!strcmp(model, "equil"))
        C_equil(energy, nFlux, params, specNum, flux, fluxErr, initStr);
    else if(!strcmp(model, "gaussianLine"))
        C_gaussianLine(energy, nFlux, params, specNum, flux, fluxErr, initStr);
    else if(!strcmp(model, "gaussDem"))
        C_gaussDem(energy, nFlux, params, specNum, flux, fluxErr, initStr);
    else if(!strcmp(model, "gnei"))
        C_gnei(energy, nFlux, params, specNum, flux, fluxErr, initStr);
    else if(!strcmp(model, "hatm"))
        C_hatm(energy, nFlux, params, specNum, flux, fluxErr, initStr);
    else if(!strcmp(model, "kerrbb"))
        C_kerrbb(energy, nFlux, params, specNum, flux, fluxErr, initStr);
    else if(!strcmp(model, "kerrdisk"))
        C_kerrdisk(energy, nFlux, params, specNum, flux, fluxErr, initStr);
    else if(!strcmp(model, "xslaor"))
        C_xslaor(energy, nFlux, params, specNum, flux, fluxErr, initStr);
    else if(!strcmp(model, "laor2"))
        C_laor2(energy, nFlux, params, specNum, flux, fluxErr, initStr);
    else if(!strcmp(model, "lorentzianLine"))
        C_lorentzianLine(energy, nFlux, params, specNum, flux, fluxErr, initStr);
    else if(!strcmp(model, "meka"))
        C_meka(energy, nFlux, params, specNum, flux, fluxErr, initStr);
    else if(!strcmp(model, "mekal"))
        C_mekal(energy, nFlux, params, specNum, flux, fluxErr, initStr);
    else if(!strcmp(model, "xsmkcf"))
        C_xsmkcf(energy, nFlux, params, specNum, flux, fluxErr, initStr);
    else if(!strcmp(model, "nei"))
        C_nei(energy, nFlux, params, specNum, flux, fluxErr, initStr);
    else if(!strcmp(model, "nlapec"))
        C_nlapec(energy, nFlux, params, specNum, flux, fluxErr, initStr);
    else if(!strcmp(model, "npshock"))
        C_npshock(energy, nFlux, params, specNum, flux, fluxErr, initStr);
    else if(!strcmp(model, "xsnteea"))
        C_xsnteea(energy, nFlux, params, specNum, flux, fluxErr, initStr);
    else if(!strcmp(model, "nthcomp"))
        C_nthcomp(energy, nFlux, params, specNum, flux, fluxErr, initStr);
    else if(!strcmp(model, "xspexrav"))
        C_xspexrav(energy, nFlux, params, specNum, flux, fluxErr, initStr);
    else if(!strcmp(model, "xspexriv"))
        C_xspexriv(energy, nFlux, params, specNum, flux, fluxErr, initStr);
    else if(!strcmp(model, "powerLaw"))
        C_powerLaw(energy, nFlux, params, specNum, flux, fluxErr, initStr);
    else if(!strcmp(model, "pshock"))
        C_pshock(energy, nFlux, params, specNum, flux, fluxErr, initStr);
    else if(!strcmp(model, "raysmith"))
        C_raysmith(energy, nFlux, params, specNum, flux, fluxErr, initStr);
    else if(!strcmp(model, "rnei"))
        C_rnei(energy, nFlux, params, specNum, flux, fluxErr, initStr);
    else if(!strcmp(model, "sedov"))
        C_sedov(energy, nFlux, params, specNum, flux, fluxErr, initStr);
    else if(!strcmp(model, "sirf"))
        C_sirf(energy, nFlux, params, specNum, flux, fluxErr, initStr);
    else if(!strcmp(model, "snapec"))
        C_snapec(energy, nFlux, params, specNum, flux, fluxErr, initStr);
    else if(!strcmp(model, "tapec"))
        C_tapec(energy, nFlux, params, specNum, flux, fluxErr, initStr);
    else if(!strcmp(model, "tbabs"))
        C_tbabs(energy, nFlux, params, specNum, flux, fluxErr, initStr);
    else
        err = 1;

    return err;
}

