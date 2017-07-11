# disrupt-xspec

### Compiling ###

First copy `Makefile.in.template` to `Makefile.in`

```bash
$ cp Makefile.in.template Makefile.in
```

Your `LD_LIBRARY_PATH` or `DYLD_LIBRARY_PATH` may have to include the Xspec's `lib/` directory.

### Running ###

The `HEADAS` environment variable must be set.  `HEADAS` points to the directory of the Xspec installation (the directory that ends with the platform name of your machine).

## Building and Running `xsmodels` Python Package ##

The `HEADAS` environment variable must be set and possibly `(DY)LD_LIBRARY_PATH`.  Navigate to the `xsmodels` directory and do an in-place build:

```bash
$ python setup.py build_ext -i
```

Now you can run the `test.py` script in the top-level directory.  It will print out the energy, params, flux, and fluxError arrays and produce a plot saved in `flux_py.png`.
