#include "isddft.h"

#include "main_RPA.h"
#include "parallelization_RPA.h"
#include "parallelization.h"

void prepare_PQ_operators(SPARC_OBJ *pSPARC, RPA_OBJ *pRPA);

int find_nearby_band_indices_gamma(SPARC_OBJ *pSPARC, int *bandStartIndices, double *allEpsilonsGamma, int *nearbyBandIndices, int *neighborBandStartEnd, int *allBandComms);

void get_neighborBands_gamma(SPARC_OBJ *pSPARC, int *nearbyBandIndices, int *neighborBandStartEnd, int *allBandComms, 
    int *neighborBandIndices, double *neighborBands);