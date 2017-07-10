#include <stdio.h>
#include <xsFortran.h>
#include <funcWrappers.h>

int main(int argc, char *argv[])
{
    FNINIT();

    printf("Hello world!\n");

    int nFlux = 10;
    double energy[nFlux+1];
    double flux[nFlux];
    double fluxError[nFlux];

    int nPar = 2;
    double params[nPar];
    params[0] = 2.0;
    params[1] = 1.0;

    double ei = 1.0;
    double ef = 10.0;
    double de = (ef-ei) / nFlux;
    
    int i;
    for(i=0; i<nFlux+1; i++)
    {
        energy[i] = ei + i*de;
    }

    printf("Energy:");
    for(i=0; i<nFlux+1; i++)
        printf(" %.3lg", energy[i]);
    printf("\n");

    C_powerLaw(energy, nFlux, params, 0, flux, fluxError, NULL);

    printf("Flux:");
    for(i=0; i<nFlux; i++)
        printf(" %.3lg", flux[i]);
    printf("\n");
    
    printf("FluxError:");
    for(i=0; i<nFlux+1; i++)
        printf(" %.3lg", fluxError[i]);
    printf("\n");

    return 0;
}
