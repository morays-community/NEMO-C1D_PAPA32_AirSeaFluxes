MODULE infmod
   !!======================================================================
   !!                       ***  MODULE  infmod  ***
   !! Machine Learning Inferences : manage connexion with external ML codes 
   !!======================================================================
   !! History :  4.2.1  ! 2023-09  (A. Barge)  Original code
   !!----------------------------------------------------------------------

   !!----------------------------------------------------------------------
   !!   naminf          : machine learning models formulation namelist
   !!   inferences_init : initialization of Machine Learning based models
   !!   inferences      : ML based models
   !!   inf_snd         : send data to external trained model
   !!   inf_rcv         : receive inferences from external trained model
   !!----------------------------------------------------------------------
   USE oce             ! ocean fields
   USE dom_oce         ! ocean domain fields
   USE sbc_oce         ! ocean surface fields
   USE inffld
   USE cpl_oasis3      ! OASIS3 coupling
   USE timing
   USE iom
   USE in_out_manager
   USE lib_mpp

   IMPLICIT NONE
   PRIVATE

   PUBLIC inf_alloc          ! function called in inferences_init 
   PUBLIC inf_dealloc        ! function called in inferences_final
   PUBLIC inferences_init    ! routine called in nemogcm.F90
   PUBLIC inferences         ! routine called in stpmlf.F90
   PUBLIC inferences_final   ! routine called in nemogcm.F90

   INTEGER, PARAMETER ::   jps_ux  = 1   ! Wind speed in x-dir
   INTEGER, PARAMETER ::   jps_uy = 2    ! Wind speed in y-dir
   INTEGER, PARAMETER ::   jps_toce = 3  ! Sea surface temperature
   INTEGER, PARAMETER ::   jps_tair  = 4 ! Air temperature
   INTEGER, PARAMETER ::   jps_p = 5     ! Sea level pressure
   INTEGER, PARAMETER ::   jps_q = 6     ! Specific humidity
   INTEGER, PARAMETER ::   jps_inf = 6   ! total number of sendings

   INTEGER, PARAMETER ::   jpr_taux = 1   ! wind stress in x-dir
   INTEGER, PARAMETER ::   jpr_tauy = 2   ! wind stress in y-dir
   INTEGER, PARAMETER ::   jpr_qs = 3     ! sensible heat
   INTEGER, PARAMETER ::   jpr_ql = 4     ! latent heat
   INTEGER, PARAMETER ::   jpr_inf = 4   ! total number of receptions

   INTEGER, PARAMETER ::   jpinf = MAX(jps_inf,jpr_inf) ! Maximum number of exchanges

   TYPE( DYNARR ), SAVE, DIMENSION(jpinf) ::  infsnd, infrcv  ! sent/received inferences

   !
   !!-------------------------------------------------------------------------
   !!                    Namelist for the Inference Models
   !!-------------------------------------------------------------------------
   !                           !!** naminf namelist **
   !TYPE ::   FLD_INF              !: Field informations ...  
   !   CHARACTER(len = 32) ::         ! 
   !END TYPE FLD_INF
   !
   LOGICAL , PUBLIC ::   ln_inf    !: activate module for inference models
   
   !!-------------------------------------------------------------------------

CONTAINS

   INTEGER FUNCTION inf_alloc()
      !!----------------------------------------------------------------------
      !!             ***  FUNCTION inf_alloc  ***
      !!----------------------------------------------------------------------
      INTEGER :: ierr
      INTEGER :: jn
      !!----------------------------------------------------------------------
      ierr = 0
      !
      DO jn = 1, jpinf
         IF( srcv(ntypinf,jn)%laction ) ALLOCATE( infrcv(jn)%z3(jpi,jpj,srcv(ntypinf,jn)%nlvl), STAT=ierr )
         IF( ssnd(ntypinf,jn)%laction ) ALLOCATE( infsnd(jn)%z3(jpi,jpj,ssnd(ntypinf,jn)%nlvl), STAT=ierr )
         inf_alloc = MAX(ierr,0)
      END DO
      !
   END FUNCTION inf_alloc

   
   INTEGER FUNCTION inf_dealloc()
      !!----------------------------------------------------------------------
      !!             ***  FUNCTION inf_dealloc  ***
      !!----------------------------------------------------------------------
      INTEGER :: ierr
      INTEGER :: jn
      !!----------------------------------------------------------------------
      ierr = 0
      !
      DO jn = 1, jpinf
         IF( srcv(ntypinf,jn)%laction ) DEALLOCATE( infrcv(jn)%z3, STAT=ierr )
         IF( ssnd(ntypinf,jn)%laction ) DEALLOCATE( infsnd(jn)%z3, STAT=ierr )
         inf_dealloc = MAX(ierr,0)
      END DO
      !
   END FUNCTION inf_dealloc


   SUBROUTINE inferences_init 
      !!----------------------------------------------------------------------
      !!             ***  ROUTINE inferences_init  ***
      !!
      !! ** Purpose :   Initialisation of the models that rely on external inferences
      !!
      !! ** Method  :   * Read naminf namelist
      !!                * create data for models
      !!----------------------------------------------------------------------
      !
      INTEGER ::   ios   ! Local Integer
      !!
      LOGICAL ::  ln_inf
      !!----------------------------------------------------------------------
      !
      ! ================================ !
      !      Namelist informations       !
      ! ================================ !
      !
      ln_inf = .TRUE.
      !
      IF( lwp ) THEN                        ! control print
         WRITE(numout,*)
         WRITE(numout,*)'inferences_init : Setting inferences models'
         WRITE(numout,*)'~~~~~~~~~~~~~~~'
      END IF
      !
      IF( ln_inf .AND. .NOT. lk_oasis )   CALL ctl_stop( 'inferences_init : External inferences coupled via OASIS, but key_oasis3 disabled' )
      !
      !
      ! ======================================== !
      !     Define exchange needs for Models     !
      ! ======================================== !
      !
      ! default definitions of ssnd snd srcv
      srcv(ntypinf,:)%laction = .TRUE.  ;  srcv(ntypinf,:)%clgrid = 'T'  ;  srcv(ntypinf,:)%nsgn = 1.
      srcv(ntypinf,:)%nct = 1
      !
      ssnd(ntypinf,:)%laction = .TRUE.  ;  ssnd(ntypinf,:)%clgrid = 'T'  ;  ssnd(ntypinf,:)%nsgn = 1.
      ssnd(ntypinf,:)%nct = 1
      
      IF( ln_inf ) THEN
      
         ! -------------------------------- !
         !      Kenigson et al. (2022)      !
         ! -------------------------------- !

         ! ssnd: Wind speed, Air Temperature, Specific humidity, Sea Surface Temperature, Sea level pressure
         ssnd(ntypinf,jps_ux)%clname =  'E_OUT_0'
         ssnd(ntypinf,jps_uy)%clname = 'E_OUT_1'
         ssnd(ntypinf,jps_toce)%clname =  'E_OUT_2'
         ssnd(ntypinf,jps_tair)%clname =  'E_OUT_3'
         ssnd(ntypinf,jps_p)%clname =  'E_OUT_4'
         ssnd(ntypinf,jps_q)%clname =  'E_OUT_5'
         ! Number of depth levels to couple
         ssnd(ntypinf,:)%nlvl = 1

         ! srcv: Wind stress, latent heat, sensible heat, vaporization heat
         srcv(ntypinf,jpr_taux)%clname = 'E_IN_0'
         srcv(ntypinf,jpr_tauy)%clname = 'E_IN_1'
         srcv(ntypinf,jpr_Qs)%clname = 'E_IN_2'
         srcv(ntypinf,jpr_Ql)%clname = 'E_IN_3'
         ! Number of depth levels to couple
         srcv(ntypinf,:)%nlvl = 1

         ! ------------------------------ !
      END IF
      ! 
      ! ================================= !
      !   Define variables for coupling
      ! ================================= !
      CALL cpl_var(jpinf, jpinf, 1, ntypinf)
      !
      IF( inf_alloc() /= 0 )     CALL ctl_stop( 'STOP', 'inf_alloc : unable to allocate arrays' )
      IF( inffld_alloc() /= 0 )  CALL ctl_stop( 'STOP', 'inffld_alloc : unable to allocate arrays' ) 
      !
   END SUBROUTINE inferences_init


   SUBROUTINE inferences( kt, wndx, wndy, tair, sst, hum, slp )
      !!----------------------------------------------------------------------
      !!             ***  ROUTINE inferences  ***
      !!
      !! ** Purpose :   update the ocean data with the ML based models
      !!
      !! ** Method  :   *  
      !!                * 
      !!----------------------------------------------------------------------
      INTEGER, INTENT(in) ::  kt               ! ocean time step
      REAL(wp), DIMENSION(jpi,jpj), INTENT(in) :: wndx, wndy, tair, hum, sst, slp ! surface fields
      !
      INTEGER :: isec, info, jn                       ! local integer
      !!----------------------------------------------------------------------
      !
      IF( ln_timing )   CALL timing_start('Python')
      !
      isec = ( kt - nit000 ) * NINT( rn_Dt )       ! Date of exchange 
      info = OASIS_idle
      !
      ! ------  Prepare data to send ------
      !
      ! Wind speed
      infsnd(jps_ux)%z3(:,:,ssnd(ntypinf,jps_ux)%nlvl) = wndx(:,:)
      infsnd(jps_uy)%z3(:,:,ssnd(ntypinf,jps_uy)%nlvl) = wndy(:,:)
      ! Air temperature
      infsnd(jps_tair)%z3(:,:,ssnd(ntypinf,jps_tair)%nlvl) = tair(:,:)
      ! Ocean temperature
      infsnd(jps_toce)%z3(:,:,ssnd(ntypinf,jps_toce)%nlvl) = sst(:,:)
      ! Specific humidity
      infsnd(jps_q)%z3(:,:,ssnd(ntypinf,jps_q)%nlvl) = hum(:,:)
      ! Sea level pressure
      infsnd(jps_p)%z3(:,:,ssnd(ntypinf,jps_p)%nlvl) = slp(:,:)
      !
      ! ========================
      !   Proceed all sendings
      ! ========================
      !
      DO jn = 1, jpinf
         IF ( ssnd(ntypinf,jn)%laction ) THEN
            CALL cpl_snd( jn, isec, ntypinf, infsnd(jn)%z3, info)
         ENDIF
      END DO
      !
      ! .... some external operations ....
      !
      ! ==========================
      !   Proceed all receptions
      ! ==========================
      !
      DO jn = 1, jpinf
         IF( srcv(ntypinf,jn)%laction ) THEN
            CALL cpl_rcv( jn, isec, ntypinf, infrcv(jn)%z3, info)
         ENDIF
      END DO
      !
      ! ------ Distribute receptions  ------
      !
      ! Store latent, sensible, vaporization heat and wind stress
      ext_ql(:,:)   = infrcv(jpr_ql)%z3(:,:,srcv(ntypinf,jpr_ql)%nlvl)
      ext_qs(:,:)   = infrcv(jpr_qs)%z3(:,:,srcv(ntypinf,jpr_qs)%nlvl)
      ext_taux(:,:) = infrcv(jpr_taux)%z3(:,:,srcv(ntypinf,jpr_taux)%nlvl)
      ext_tauy(:,:) = infrcv(jpr_tauy)%z3(:,:,srcv(ntypinf,jpr_tauy)%nlvl)
      ! Write returned results
      CALL iom_put( "ext_ql" , ext_ql )
      CALL iom_put( "ext_qs" , ext_qs )
      CALL iom_put( "ext_taux" , ext_taux )
      CALL iom_put( "ext_tauy" , ext_tauy )
      !
      IF( ln_timing )   CALL timing_stop('Python')
      !
   END SUBROUTINE inferences


   SUBROUTINE inferences_final
      !!----------------------------------------------------------------------
      !!             ***  ROUTINE inferences_final  ***
      !!
      !! ** Purpose :   Free memory used for inferences modules
      !!
      !! ** Method  :   * Deallocate arrays
      !!----------------------------------------------------------------------
      !
      IF( inf_dealloc() /= 0 )     CALL ctl_stop( 'STOP', 'inf_dealloc : unable to free memory' )
      IF( inffld_dealloc() /= 0 )  CALL ctl_stop( 'STOP', 'inffld_dealloc : unable to free memory' )      
      !
   END SUBROUTINE inferences_final 
   !!=======================================================================
END MODULE infmod
