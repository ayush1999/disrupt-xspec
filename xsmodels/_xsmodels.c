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

//static char kerrbb_doctring[] = 
//    "Calculate a spectrum with the Xspec kerrbb model.";

static PyObject *xsmodels_powerLaw(PyObject *self, PyObject *args);
//static PyObject *xsmodels_kerrbb(PyObject *self, PyObject *args);

static PyMethodDef xsmodels_methods[] = {
    {"powerLaw", xsmodels_powerLaw, METH_VARARGS, powerLaw_docstring},
//    {"kerrbb", xsmodels_kerrbb, METH_VARARGS, kerrbb_docstring},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef xsmodels = {
    PyModuleDef_HEAD_INIT,
    "xsmodels",  /* Module Name*/
    xsmodels_docstring, /* Module docstring */
    -1, 
    xsmodels_methods
};

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
