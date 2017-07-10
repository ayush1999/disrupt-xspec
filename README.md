# disrupt-xspec

### Compiling ###

First copy `Makefile.in.template` to `Makefile.in`

```bash
$ cp Makefile.in.template Makefile.in
```

Your `LD_LIBRARY_PATH' or `DYLD_LIBRARY_PATH' may have to include the Xspec's `lib/' directory.

### Running ###

The `HEADAS' environment variable must be set, and the `heainit' has to be run.  `HEADAS' points to the directory of the Xspec installation (the directory that ends with the platform name of your machine).  The `hea-init.sh' script is found in that directory as well.
