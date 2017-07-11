#include <stdio.h>
#include <math.h>
#include <xsFortran.h>
#include <funcWrappers.h>

int main(int argc, char *argv[])
{
    FNINIT();

    char fname[] = "flux.txt";

    printf("Hello world!\n");

    int nFlux = 999;
    double energy[nFlux+1];
    double flux[nFlux];
    double fluxError[nFlux];

    
    /*
    int nPar = 2;
    double params[nPar];
    params[0] = 2.0; // power-law index
    params[1] = 1.0; // normalization
    */
    
    int nPar = 10;
    double params[nPar];
    params[0] = 0.0; // eta (0.0 == no-torque ISCO)
    params[1] = 0.9; // a_star (-1< a_star < 1
    params[2] = 60.0; // i (inclination, degrees, <85)
    params[3] = 10.0; // M_BH (M_solar)
    params[4] = 1.0; // M_dot effective (10^18 g/s)
    params[5] = 10.0; // D_L (kpc)
    params[6] = 1.7; // spectral hardening (1.5-1.9)
    params[7] = 1.0; // self irradiation flag
    params[8] = 0.0; // limb darkening flag
    params[9] = 1.0; // normalization

    double ei = 0.01;
    double ef = 10.0;
    double de = (ef-ei) / nFlux;
    
    int i;
    for(i=0; i<nFlux+1; i++)
        energy[i] = ei * pow(ef/ei, ((double) i)/nFlux);

    FILE *f = fopen(fname, "w");
    fprintf(f, "Energy:");
    for(i=0; i<nFlux+1; i++)
        fprintf(f, " %.3lg", energy[i]);
    fprintf(f, "\n");
    fclose(f);

    //C_powerLaw(energy, nFlux, params, 0, flux, fluxError, NULL);
    C_kerrbb(energy, nFlux, params, 0, flux, fluxError, NULL);

    f = fopen(fname, "a");
    fprintf(f, "Flux:");
    for(i=0; i<nFlux; i++)
        fprintf(f, " %.3lg", flux[i]);
    fprintf(f, "\n");
    
    fprintf(f, "FluxError:");
    for(i=0; i<nFlux+1; i++)
        fprintf(f, " %.3lg", fluxError[i]);
    fprintf(f, "\n");
    fclose(f);

    return 0;
}
