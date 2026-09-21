# Choose whether to use MKL or ScaLAPACK or neither
# Warning: don't turn on USE_MKL and USE_SCALAPACK simultaneously
# Set USE_MKL = 1 to compile with MKL BLAS, LAPACK, and ScaLAPACK
# Set USE_MKL = 0 otherwise
USE_MKL       = 0
# Set USE_SCALAPACK = 1 to compile with non-MKL BLAS, LAPACK, and ScaLAPACK
# Set USE_SCALAPACK = 0 to compile with non-MKL BLAS and LAPACK only
USE_SCALAPACK = 1
# Set USE_DP_SUBEIG = 1 to use SPARC rather than ScaLAPACK routines for matrix data distribution
# (USE_DP_SUBEIG = 1 is required if both USE_MKL = 0 and USE_SCALAPACK = 0)
# Set USE_DP_SUBEIG = 0 to use ScaLAPACK rather than SPARC routines
USE_DP_SUBEIG = 0
# Set USE_FFTW = 1 to use FFTW for fast Fourier transform in vdWDF. Don't open it together with USE_MKL
USE_FFTW      = 0
# Set DEBUG_MODE = 1 to run with debug mode and print debug output
DEBUG_MODE    = 0

# Set USE_SOCKET = 0 to disable compilation with socket support
USE_SOCKET    = 0

# set USE_ELPA = 1 to use ELPA for parallel eigensolver. ELPA requires MKL or Scalapack
USE_ELPA      = 0

# Enable SIMD vectorization for complex stencil routines
# CAUTION: for some compilers this results in wrong results! Use for intel/19.0.3 or later versions
ENABLE_SIMD_COMPLEX = 1

# Enable ACCEL by GT
USE_ACCELGT = 0

USE_FRONTIER = 1

# Specify the path MKLROOT if it's not already set to compile with MKL, e.g,
# MKLROOT = /opt/intel/compilers_and_libraries_2017.4.196/linux/mkl

# Specify the path to ScaLAPACK, LAPACK and BLAS if necessary, and 
# add to LDFLAGS. Note that sometimes LDFLAGS already contains the default 
# path to these libraries, or the libraries are located in the default search
# path. In those cases, the following is not needed.
# SCALAPACKROOT = /nv/hp27/qxu78/data/scalapack-2.0.2
# LDFLAGS += -L$(SCALAPACKROOT)
# LAPACKROOT = /usr/local/pacerepov1/lapack/3.6.0
# LDFLAGS += -L$(LAPACKROOT)
# BLASROOT = /usr/lib64
# LDFLAGS += -L$(BLASROOT)

CPPFLAGS = -Iinclude/ -Ixc/exx/include -Ixc/vdW/d3/include -Ixc/vdW/vdWDF/include -Ixc/mgga/include -Ixc/rpa/include -IlinearScaling/include -Iofdft/ -Icyclix/include -Icyclix/mlff -IhighT/mlff -Imlff/ -IhighT/include \
           -Iatom/include -Iatom/mgga_atom/include -Iatom/exx_atom/include -Ixc/hubbard/include
ifeq ($(USE_FRONTIER), 1)
CPPFLAGS += -I/opt/cray/pe/libsci/24.11.0/GNU/12.3/x86_64/include
CPPFLAGS += -DTIOGA
LDFLAGS += -L/opt/cray/pe/libsci/24.11.0/GNU/12.3/x86_64/lib
endif
LDLIBS   = -lrt
ifeq ($(USE_MKL), 1)
CPPFLAGS += -m64 -I${MKLROOT}/include -DUSE_MKL
LDFLAGS   = -L${MKLROOT}/lib/intel64
LDLIBS   += -Wl,-rpath=${MKLROOT}/lib/intel64,--no-as-needed -lmkl_scalapack_lp64 -lmkl_cdft_core -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lmkl_blacs_intelmpi_lp64 -lpthread -lm -ldl
else ifeq ($(USE_SCALAPACK), 1)
CPPFLAGS += -DUSE_SCALAPACK
ifeq ($(USE_FRONTIER), 0)
        LDLIBS   += -lscalapack
endif
endif

# if MKL is not used, link BLAS and LAPACK directly
ifeq ($(USE_MKL), 0)
# if you are using BLAS instead of OpenBLAS, change -lopenblas to -lblas
# and move it to after -llapack
ifeq ($(USE_FRONTIER), 0)
        LDLIBS += -lopenblas -lm
endif
ifeq ($(USE_FRONTIER), 1)
        LDLIBS += -lsci_gnu_mpi -lsci_gnu -lm
endif
#LDLIBS += -llapacke -llapack -lblas -lm
endif

# if FFTW is used, link fftw
ifeq ($(USE_FFTW), 1)
CPPFLAGS += -DUSE_FFTW
LDLIBS += -lfftw3_mpi -lfftw3 # fftw needs to be loaded before compile
endif

# To use domain parallelization + LAPACK for solving sobspace eigen problem
ifeq ($(USE_DP_SUBEIG), 1)
CPPFLAGS += -DUSE_DP_SUBEIG
endif

# to compile with DEBUG mode
ifeq ($(DEBUG_MODE), 1)
CPPFLAGS += -Wall -g -DDEBUG # -Wno-format-truncation -Wno-stringop-truncation
endif

# to enable SIMD for complex stencil routines
ifeq ($(ENABLE_SIMD_COMPLEX), 1)
CPPFLAGS += -DENABLE_SIMD_COMPLEX
endif

# to enable ELPA for parallel eigensolver
ifeq ($(USE_ELPA), 1)
CPPFLAGS += -I${ELPA_ROOT}/include/elpa_openmp-2021.11.001 -DUSE_ELPA
LDFLAGS  += -L${ELPA_ROOT}/lib
LDLIBS   += -lelpa_openmp 
endif

# for old Intel compiler, use -qopenmp instead of -fopenmp. ICC 17 and later also accepts -fopenmp. 
CFLAGS = -std=gnu99 -O3 -fopenmp

OBJSC = main.o initialization.o readfiles.o atomdata.o parallelization.o relax.o tools.o md.o      \
        electrostatics.o electronicGroundState.o electronDensity.o orbitalElecDensInit.o           \
        occupation.o gradVecRoutines.o gradVecRoutinesKpt.o nlocVecRoutines.o     \
        hamiltonianVecRoutines.o lapVecRoutines.o lapVecRoutinesKpt.o \
        linearSolver.o mixing.o exchangeCorrelation.o eigenSolver.o eigenSolverKpt.o energy.o      \
        forces.o stress.o pressure.o finalization.o spinOrbitCoupling.o printing.o \
        linearAlgebra.o kroneckerLaplacian.o mpi_nbr_shim.o\
        xc/vdW/d3/d3correction.o xc/vdW/d3/d3findR0ab.o xc/vdW/d3/d3copyC6.o                       \
        xc/vdW/d3/d3initialization.o xc/vdW/d3/d3finalization.o xc/vdW/d3/d3forceStress.o          \
        xc/vdW/vdWDF/vdWDFinitialization.o xc/vdW/vdWDF/vdWDFfinalization.o                        \
        xc/vdW/vdWDF/vdWDFnonlocalCorre.o xc/vdW/vdWDF/vdWDFreadKernel.o                 \
        xc/vdW/vdWDF/vdWDFstress.o xc/vdW/vdWDF/vdWDFparallelization.o                        \
        xc/mgga/mGGAscan.o xc/mgga/mGGArscan.o xc/mgga/mGGAr2scan.o xc/mgga/mGGAhamiltonianTerm.o  \
        xc/mgga/mGGAinitialization.o xc/mgga/mGGAfinalization.o xc/mgga/mGGAtauTransferTauVxc.o  \
        xc/mgga/mGGAstress.o \
        xc/exx/exactExchange.o xc/exx/exactExchangeKpt.o xc/exx/exactExchangeInitialization.o      \
        xc/exx/exactExchangeFinalization.o xc/exx/exactExchangeStress.o                            \
        xc/exx/exactExchangePressure.o xc/exx/exactExchangeEnergyDensity.o \
        xc/rpa/main_RPA.o xc/rpa/initialization_SPARC.o xc/rpa/initialization_RPA.o xc/rpa/generateKgrid.o xc/rpa/parallelization_RPA.o xc/rpa/parallelization_SPARC.o \
	xc/rpa/restoreElectronicGroundState.o xc/rpa/prepare_PQ_operators.o xc/rpa/collectOrbitals.o xc/rpa/subspaceIter.o xc/rpa/subspaceIterSQ.o xc/rpa/lanczosQuadrature.o xc/rpa/eigenSolverGamma_RPA.o xc/rpa/eigenSolverKpt_RPA.o \
        xc/rpa/nuChi0VecRoutines.o xc/rpa/sternheimerEquation.o xc/rpa/sternheimerEquationKpt.o xc/rpa/linearSolvers.o xc/rpa/electrostatics_RPA.o xc/rpa/printResult.o \
        xc/rpa/finalization_RPA.o xc/rpa/test_Hx_nuChi0_eigSolver.o xc/rpa/tools_RPA.o xc/rpa/traceSolverSQ.o \
        linearScaling/sq.o linearScaling/sqInitialization.o linearScaling/sqFinalization.o linearScaling/sqEnergy.o\
        linearScaling/sqDensity.o linearScaling/sqNlocVecRoutines.o linearScaling/sqParallelization.o linearScaling/sqProperties.o\
        highT/sqHighTExactExchange.o highT/sqHighT.o highT/sqHighTDensity.o highT/sqHighTInitialization.o\
        cyclix/cyclix_tools.o cyclix/cyclix_forces.o cyclix/cyclix_gradVec.o cyclix/cyclix_lapVec.o \
        cyclix/cyclix_stress.o \
        ofdft/ofdft.o\
        mlff/covariance_matrix.o mlff/regression.o mlff/spherical_harmonics.o mlff/soap_descriptor.o mlff/sparc_mlff_interface.o \
        mlff/bessel_NR.o mlff/nrutils.o mlff/sparsification.o mlff/tools_mlff.o mlff/mlff_read_write.o \
        mlff/descriptor.o mlff/ddbp_tools.o highT/mlff/internal_energy_model.o cyclix/mlff/cyclix_mlff_tools.o \
        mlff/hnl_soap.o \
        atom/sparcAtom.o atom/atomicStates.o atom/electronicGroundStateAtom.o atom/finalizeAtom.o atom/initializationAtom.o \
        atom/spectralFunctionsAtom.o atom/toolsAtom.o atom/mgga_atom/mGGApotentialAtom.o atom/mgga_atom/mGGAtoolsAtom.o \
        atom/exx_atom/exxToolsAtom.o atom/exx_atom/exxPotentialEnergyAtom.o \
        xc/hubbard/hubbardInitialization.o xc/hubbard/locOrbRoutines.o xc/hubbard/occupationMatrix.o \
        xc/hubbard/hubbardForce.o xc/hubbard/hubbardStress.o xc/hubbard/hubbardPressure.o

# Switch the socket compilation
# For future developers: please keep the socket options separate from the
# main objects
# Please do not add socket related *.o files manually in OBJSC
ifeq ($(USE_SOCKET), 1)
CPPFLAGS += -Isocket/ -DUSE_SOCKET
OBJSC := $(patsubst main.o,main_socket.o,$(OBJSC))
OBJSC += socket/driver.o socket/libinetsocket.o socket/libunixsocket.o
endif



LIBBASE = ../lib/sparc
TESTBASE = ../.ci

override CC=mpicc

all: sparc

# Note the implicit rule to compile '.c' files into '.o' files is
# %.o : %.c
# 	$(CC) -c $(CFLAGS) $(CPPFLAGS) $< -o $@

sparc: $(OBJSC)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $(LIBBASE) $^ $(LDLIBS)

.PHONY: clean
clean:
	rm -f  $(OBJSC) $(LIBBASE)
	rm -f socket/*.o
test: ../tests/SPARC_testing_script.py
	cd ../tests; python SPARC_testing_script.py
