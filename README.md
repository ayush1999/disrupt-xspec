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

In order to build the python package, you can now do:

```bash
$ python setup.py install
```

Note: if your python packages are located somewhere that requires sudo access, you might want to use `sudo -E` in order to preserve the environment variables set above.

You should run the tests to make sure the package built and installed correctly. Note: you need to have the `pytest` package installed. In the top level directory, type

```bash
$ py.test
```

This will collect and run all tests. Ideally, no tests should throw errors. If any do, something went wrong with the installation.

