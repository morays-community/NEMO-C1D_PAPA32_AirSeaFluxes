MODULE inffld
   !!======================================================================
   !!                       ***  MODULE inffld  ***
   !! Inferences module :   variables defined in core memory
   !!======================================================================
   !! History :  4.2  ! 2023-09  (A. Barge)  Original code
   !!----------------------------------------------------------------------

   !!----------------------------------------------------------------------
   !!   inffld_alloc : allocation of fields arrays for inferences module (infmod)
   !!----------------------------------------------------------------------        
   !!=====================================================
   USE par_oce        ! ocean parameters
   USE lib_mpp        ! MPP library

   IMPLICIT NONE
   PRIVATE

   PUBLIC   inffld_alloc   ! routine called in infmod.F90
   PUBLIC   inffld_dealloc ! routine called in infmod.F90

   !!----------------------------------------------------------------------
   !!                    2D Inference Module fields
   !!----------------------------------------------------------------------
   REAL(wp), PUBLIC, ALLOCATABLE, SAVE, DIMENSION(:,:)  :: ext_qs, ext_ql, ext_taux, ext_tauy !: fields to store 2D Python fields

   !!----------------------------------------------------------------------
   !!                    3D Inference Module fields
   !!----------------------------------------------------------------------
   !!REAL(wp), PUBLIC, ALLOCATABLE, SAVE, DIMENSION(:,:,:)  :: dTdt, dSdt

CONTAINS

   INTEGER FUNCTION inffld_alloc()
      !!---------------------------------------------------------------------
      !!                  ***  FUNCTION inffld_alloc  ***
      !!---------------------------------------------------------------------
      INTEGER :: ierr
      !!---------------------------------------------------------------------
      ierr = 0
      !
      ALLOCATE( ext_taux(jpi,jpj), ext_tauy(jpi,jpj), ext_qs(jpi,jpj), ext_ql(jpi,jpj) , STAT=ierr )
      inffld_alloc = ierr
      !
   END FUNCTION

   
   INTEGER FUNCTION inffld_dealloc()
      !!---------------------------------------------------------------------
      !!                  ***  FUNCTION inffld_dealloc  ***
      !!---------------------------------------------------------------------
      INTEGER :: ierr
      !!---------------------------------------------------------------------
      ierr = 0
      !
      DEALLOCATE( ext_taux, ext_tauy, ext_qs, ext_ql  , STAT=ierr )
      inffld_dealloc = ierr
      !
   END FUNCTION

END MODULE inffld
