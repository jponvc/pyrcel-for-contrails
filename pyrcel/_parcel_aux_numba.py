import numba as nb
import numpy as np
from numba.pycc import CC

import pyrcel.constants as c

## Define double DTYPE
DTYPE = np.float64

PI = 3.14159265358979323846264338328
N_STATE_VARS = c.N_STATE_VARS

# AOT/numba stuff
auxcc = CC("parcel_aux_numba")
auxcc.verbose = True


## Auxiliary, single-value calculations with GIL released for derivative
## calculations
@nb.njit()
@auxcc.export("sigma_w", "f8(f8)")
def sigma_w(T):
    """See :func:`pyrcel.thermo.sigma_w` for full documentation"""
    return 0.0761 - (1.55e-4) * (T - 273.15)


@nb.njit()
@auxcc.export("ka", "f8(f8, f8, f8)")
def ka(T, r, rho):
    """See :func:`pyrcel.thermo.ka` for full documentation"""
    ka_cont = 1e-3 * (4.39 + 0.071 * T)
    denom = 1.0 + (ka_cont / (c.at * r * rho * c.Cp)) * np.sqrt(
        (2 * PI * c.Ma) / (c.R * T)
    )
    return ka_cont / denom


@nb.njit()
@auxcc.export("dv", "f8(f8, f8, f8, f8)")
def dv(T, r, P, accom):
    """See :func:`pyrcel.thermo.dv` for full documentation"""
    P_atm = P * 1.01325e-5  # Pa -> atm
    dv_cont = 1e-4 * (0.211 / P_atm) * ((T / 273.0) ** 1.94)
    denom = 1.0 + (dv_cont / (accom * r)) * np.sqrt((2 * PI * c.Mw) / (c.R * T))
    return dv_cont / denom


@nb.njit()
@auxcc.export("es", "f8(f8)")
def es(T):
    """See :func:`pyrcel.thermo.es` for full documentation"""
    return 611.2 * np.exp(17.67 * T / (T + 243.5))


@nb.njit()
@auxcc.export("Seq", "f8(f8, f8, f8)")
def Seq(r, r_dry, T, kappa):
    """See :func:`pyrcel.thermo.Seq` for full documentation."""
    A = (2.0 * c.Mw * sigma_w(T)) / (c.R * T * c.rho_w * r)
    B = 1.0
    if kappa > 0.0:
        B = (r**3 - (r_dry**3)) / (r**3 - (r_dry**3) * (1.0 - kappa))
    return np.exp(A) * B - 1.0

## RHS Derivative callback function
@nb.njit(parallel=True)
@auxcc.export("parcel_ode_sys", "f8[:](f8[:], f8, i4, f8[:], f8[:], f8, f8[:], f8)")
def parcel_ode_sys(y, t, nr, r_drys, Nis, V, kappas, accom):
    """Calculates the instantaneous time-derivative of the parcel model system.

    Given a current state vector `y` of the parcel model, computes the tendency
    of each term including thermodynamic (pressure, temperature, etc) and aerosol
    terms. The basic aerosol properties used in the model must be passed along
    with the state vector (i.e. if being used as the callback function in an ODE
    solver).

    Parameters
    ----------
    y : array_like
        Current state of the parcel model system,
            * y[0] = altitude, m
            * y[1] = Pressure, Pa
            * y[2] = temperature, K
            * y[3] = water vapor mass mixing ratio, kg/kg
            * y[4] = cloud liquid water mass mixing ratio, kg/kg
            * y[5] = cloud ice water mass mixing ratio, kg/kg
            * y[6] = parcel supersaturation
            * y[7:] = aerosol bin sizes (radii), m
    t : float
        Current simulation time, in seconds.
    nr : Integer
        Number of aerosol radii being tracked.
    r_drys : array_like
        Array recording original aerosol dry radii, m.
    Nis : array_like
        Array recording aerosol number concentrations, 1/(m**3).
    V : array containing variables:
        T_a : ambient temperature, in Kelvin
        P0 : total atmospheric pressure, in Pa
        RHi : ambient ice saturation ratio
        G : gradient of the contrail mixing line (see B Kärcher et al., 2015), in Pa/K
        T_e: exhaust temperature, in Kelvin
    kappas : array_like
        Array recording aerosol hygroscopicities.
    accom : float, optional (default=:const:`constants.ac`)
        Condensation coefficient.

    Returns
    -------
    x : array_like
        Array of shape (``nr``+7, ) containing the evaluated parcel model
        instaneous derivative.

    Notes
    -----
    This function is implemented using numba; it does not need to be just-in-
    time compiled in order ot function correctly, but it is set up ahead of time
    so that the internal loop over each bin growth term is parallelized.

    """
    z = y[0]
    P = y[1]
    T = y[2]
    wv = y[3]
    wc = y[4]
    wi = y[5]
    S = y[6]
    rs = y[N_STATE_VARS:N_STATE_VARS+nr]

    T_c = T - 273.15  # convert temperature to Celsius
    pv_sat = es(T_c)  # saturation vapor pressure
    Tv = (1.0 + 0.61 * wv) * T
    e = (1.0 + S) * pv_sat  # water vapor pressure

    def p_i_0(T): 
        """Calculates the saturation vapour pressure above a flat surface of ice.

        Parameters
        ----------
        T : array_like

        Returns
        -------
        x : array_like
            Saturation vapour pressure above a flat surface of ice according to Murphy and Koop, 2005
        """   
        return np.exp(9.550426 - 5723.265/T + 3.53068 * np.log(T) - 0.00728332*T) 

    ## Compute air densities from current state
    rho_air = P / c.Rd / Tv
    #: TODO - port to parcel.py
    rho_air_dry = (P - e) / c.Rd / T

    ## Begin computing tendencies
    dP_dt = 0.0 # assume contrail formation is isenthalpic and occurs at a fixed altitude (and total atmospheric pressure)
    dwc_dt = 0.0

    drs_dt = np.empty_like(rs)
    dNis_dt = np.empty_like(Nis)

    ## Set boundary conditions for the contrail mixing process using the labile parameter "V". Temperature of exhaust (T_e, in Kelvin) and ambient air (T_a, in Kelvin) and total atmospheric pressure (P0, in Pa)
    T_e = V[4] 
    T_a = V[0]
    P0 = V[1]

    ## Define parameters used to describe contrail dilution using one of two methods:

    ## (1) according to B Kärcher et al., 2015
    tau_m = 10e-3 # timescale over which contrail mixing is unaffected by entrainment, in seconds
    beta = 0.9 # plume dilution parameter

    ## (2) according to U Schumann et al., 1998 (used in Schuman et al., 2012)
    ## tau_m = 10 ** (np.log10((70/7000))/0.8)
    ## beta = 0.8
    
    expansion_time = tau_m * ((T_e - T_a) / (T - T_a)) ** (1/beta) 
    
    if expansion_time > tau_m:
        dilution_parameter = (tau_m/expansion_time) ** beta
    else:
        dilution_parameter = 1
    
    dT_dt = - beta * (T_e - T_a)/tau_m * dilution_parameter ** (1 + 1/beta) # rate of change of plume temperature, in Kelvin

    for i in nb.prange(nr):
        r = rs[i]
        r_dry = r_drys[i]
        kappa = kappas[i]
        Ni = Nis[i]

        ## Reduce number concentration within a given bin according to the chosen dilution parameter (see above), discriminating between the dilution of exhaust emissions and entrainment of ambient particles
        if kappa == 0.500001: # method to discriminate between ambient particles and aircraft emissions (to update)
            Ni = T_a/T * (1 - dilution_parameter) * Nis[i]
        else:
            Ni = dilution_parameter * (rho_air_dry)/((P0 * 28.96e-3) / (8.3145 * T_e)) * Nis[i] 

        Ni = max(Ni, 0) # prevent negative bin concentrations

        ## Non-continuum diffusivity/thermal conductivity of air near particle
        dv_r = dv(T, r, P, accom)
        ka_r = ka(T, r, rho_air)

        ## Determine conditions for a droplet to nucleate ice homogeneously
    
        def J_Koop_new(T, r, kappa, r_dry):
            """Calculates the homogeneous freezing rate for a given temperature, wet and dry droplet radius, and kappa value. This approach parameterizes the water activity according to kappa-Köhler theory
            (Petters and Kreidenweis, 2007) and uses this in combination with the parameterization for homogeneous ice nucleation proposed by T Koop et al., 2000. I have also introduced a modified water 
            activity using the results in C Marcolli, 2020, that result in a reduced freezing temperature for nanodroplets. The overall approach has previously been used by D Lewellen, 2020.
 
            Parameters
            ----------
            T : float
                Plume temperature, in Kelvin
            r : float
                Wet particle radii, in m
            kappa : float
                Particle hygroscopicity, no units
            r_dry : float
                Dry particle radii, in m

            Returns
            -------
            x : J
                Homogeneous freezing rate, in units of: ice nucleation events per unit volume per unit time

            """   
            ## Parameterize the surface tension
            T_c = 647.096
            tau = 1 - T/T_c
            mu = 1.256
            B_ = 0.2358
            b_ = -0.625

            gamma_vw = B_ * tau ** mu * (1 + b_ * tau)

            ## Specify the input conditions
            r_w = r
            T = T
            P0 = 1e5/1e6 # 101325 ~ 1e5 normalised by 1e6 (for MPa)
            P_droplet = (P0 + (2 * gamma_vw / r_w)/1e6) / 1e3 # converting to GPa
            
            ## Apply the homogeneous freezing rate parameterization
            a_w = (r ** 3 - r_dry ** 3)/(r **3 - r_dry ** 3 * (1-kappa))

            u_w_i_minus_u_w_0 = 210368 + 131.438 * T - 3.32373e6/T - 41729.1 * np.log(T)
            a_w_i = np.exp(u_w_i_minus_u_w_0/(8.3145 * T))

            v_w_0 = -230.76 - 0.1478 * T + 4099.2/T + 48.8341 * np.log(T)
            v_i_0 = 19.43 - 2.2e-3 * T + 1.08e-5 * T ** 2

            v_w_minus_v_i = v_w_0 * (P_droplet - 0.5 * 1.6 * P_droplet ** 2 - 1/6 * -8.8 * P_droplet ** 3) - v_i_0 * (P_droplet - 0.5 * 0.22 * P_droplet ** 2 - 1/6 * -0.17 * P_droplet **3)

            delta_a_w = a_w * np.exp(v_w_minus_v_i/(8.3145 * T)) - a_w_i

            J = 10 ** (-906.7 + 8502 * delta_a_w - 26924 * delta_a_w ** 2 + 29180 * delta_a_w ** 3)
            
            # if delta_a_w > 0.34 or delta_a_w < 0.26:
            #     return np.nan

            return J

        def frozen_fraction(T, r_dry, r_wet, kappa, dT_dt):
            """Estimates the fraction of droplets frozen under a given set of conditions. This approach is taken from B Kärcher et al., 2015. 
 
            Parameters
            ----------
            T : float
                Plume temperature, in Kelvin
            r_dry : float
                dry particle radii, in m
            r_wet : float
                Wet particle radii, in m                
            kappa : float
                Particle hygroscopicity, no units
            dT_dt: float
                Instantaneous cooling rate, in K/s

            Returns
            -------
            lmbda : fraction frozen, in arbitrary units.

            """   
            J = J_Koop_new(T, r_wet, kappa, r_dry) * 1e6

            ## Determine rate of change of the homogeneous ice nucleation rate (dJ_dT)
            dT = 0.001
            dJ_dT = (np.log(J_Koop_new(T + dT, r_wet, kappa, r_dry)) - np.log(J_Koop_new(T, r_wet, kappa, r_dry)))/(2 * dT) 
            inverse_freezing_timescale = dJ_dT * dT_dt
            
            ## Determine the frozen fraction via the liquid water volume
            volume_of_dry_particle = 4/3 * np.pi * r_dry ** 3 # volume of dry particle without the soluble material included
            volume_of_wet_particle = 4/3 * np.pi * r_wet ** 3 # volume of wet particle including water added during hygroscopic growth
            LWV = volume_of_wet_particle - volume_of_dry_particle # total volume of metastable liquid (original coating + water)
            lmbda = 1 - np.exp(-LWV * J * inverse_freezing_timescale) # determine the frozen fraction

            return lmbda

        ff = frozen_fraction(T, r_dry, r, kappa, dT_dt)

        if ff >= 1:

            ## Provided the droplet has frozen, determine the contribution of ice to dwc_dt. 
            rho_i = 916.8 # density of ice, kg/m^3
            Li = 3.335e5 # latent heat of freezing, J/kg

            # Condensation coefficient
            G_a = (rho_i * c.R * T) / (p_i_0(T) * dv_r * c.Mw)
            G_b = (Li * rho_i * ((Li * c.Mw / (c.R * T)) - 1.0)) / (ka_r * T)
            G = 1.0 / (G_a + G_b)

            ## Converting water supersaturation to ice supersaturation according to Korolev and Mazin, 2003 (not necessary to correct for particle equilibrium saturation)
            xi = es(T-273.15)/p_i_0(T) 
            delta_S = xi * S + xi - 1

            ## Size and liquid water tendencies
            dr_dt = (G / r) * delta_S
            dwc_dt += (
                rho_i * Ni * r * r * dr_dt
            )  # Contribution to liq. water tendency due to growth
            drs_dt[i] = dr_dt
    
        else:

            ## Provided the droplet has frozen, determine the contribution of liquid droplets to dwc_dt. 
            # Condensation coefficient
            G_a = (c.rho_w * c.R * T) / (pv_sat * dv_r * c.Mw)
            G_b = (c.L * c.rho_w * ((c.L * c.Mw / (c.R * T)) - 1.0)) / (ka_r * T)
            G = 1.0 / (G_a + G_b)

            ## Difference between ambient and particle equilibrium supersaturation
            Seq_r = Seq(r, r_dry, T, kappa)
            delta_S = S - Seq_r

            ## Size and liquid water tendencies
            dr_dt = (G / r) * delta_S
            dwc_dt += (
                c.rho_w * Ni * r * r * dr_dt
            )  # Contribution to liq. water tendency due to growth
            drs_dt[i] = dr_dt
            # dNis_dt[i] = dN_dt
        
    dwc_dt *= 4.0 * PI / rho_air_dry  # Hydrated aerosol size -> water mass
    dwi_dt = 0.0

    ## MASS BALANCE CONSTRAINT
    dwv_dt = -1.0 * dwc_dt 
    dz_dt = 0  

    ## GHAN (2011)

    ## Redefining "alpha" (the generation of supersaturation) according to B Kärcher et al., 2015.

    def Smw(T): # dS/dT along the contrail mixing plume
        return (p_i_0(T_a) * V[2] + V[3] * (T - T_a))/es(T-273.15)

    dT = 1e-10
    alpha = (Smw(T + dT) - Smw(T - dT))/(2 * dT) * dT_dt

    ## Using the preexisting "gamma" function

    gamma = (P * c.Ma) / (c.Mw * pv_sat)
    dS_dt = alpha - gamma * dwc_dt

    x = np.empty_like(y)
    x[0] = dz_dt
    x[1] = dP_dt
    x[2] = dT_dt
    x[3] = dwv_dt
    x[4] = dwc_dt
    x[5] = dwi_dt
    x[6] = dS_dt
    x[N_STATE_VARS:N_STATE_VARS+nr] = drs_dt[:]

    return x
