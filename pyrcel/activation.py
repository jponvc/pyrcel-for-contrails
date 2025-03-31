""" Collection of activation parameterizations.

"""
import numpy as np
from scipy.special import erfc

from . import constants as c
from .thermo import dv, dv_cont, es, ka_cont, kohler_crit, sigma_w


def _unpack_aerosols(aerosols):
    """Convert a list of :class:`AerosolSpecies` into lists of aerosol properties.

    Parameters
    ----------
    aerosols : list of :class:`AerosolSpecies`

    Returns
    -------
    dictionary of lists of aerosol properties

    """

    species, mus, sigmas, kappas, Ns = [], [], [], [], []
    for a in aerosols:
        species.append(a.species)
        mus.append(a.distribution.mu)
        sigmas.append(a.distribution.sigma)
        Ns.append(a.distribution.N)
        kappas.append(a.kappa)

    species = np.asarray(species)
    mus = np.asarray(mus)
    sigmas = np.asarray(sigmas)
    kappas = np.asarray(kappas)
    Ns = np.asarray(Ns)

    return dict(species=species, mus=mus, sigmas=sigmas, Ns=Ns, kappas=kappas)


def lognormal_activation(smax, mu, sigma, N, kappa, sgi=None, T=None, approx=True):
    """Compute the activated number/fraction from a lognormal mode

    Parameters
    ----------
    smax : float
        Maximum parcel supersaturation
    mu, sigma, N : floats
        Lognormal mode parameters; ``mu`` should be in meters
    kappa : float
        Hygroscopicity of material in aerosol mode
    sgi :float, optional
        Modal critical supersaturation; if not provided, this method will
        go ahead and compute them, but a temperature ``T`` must also be passed
    T : float, optional
        Parcel temperature; only necessary if no ``sgi`` was passed
    approx : boolean, optional (default=False)
        If computing modal critical supersaturations, use the approximated
        Kohler theory

    Returns
    -------
    N_act, act_frac : floats
        Activated number concentration and fraction for the given mode

    """

    if not sgi:
        assert T
        _, sgi = kohler_crit(T, mu, kappa, approx)

    ui = 2.0 * np.log(sgi / smax) / (3.0 * np.sqrt(2.0) * np.log(sigma))
    N_act = 0.5 * N * erfc(ui)
    act_frac = N_act / N

    return N_act, act_frac


def binned_activation(Smax, T, rs, aerosol, T_a, P0, T_e, approx=False):
    """Compute the activation statistics of a given aerosol, its transient
    size distribution, and updraft characteristics. Following Nenes et al, 2001
    also compute the kinetic limitation statistics for the aerosol.

    Parameters
    ----------
    Smax : float
        Environmental maximum supersaturation.
    T : float
        Plume temperature, in Kelvin.
    rs : array of floats
        Wet radii of aerosol/droplet population.
    aerosol : :class:`AerosolSpecies`
        The characterization of the dry aerosol.
    T_a : float
        The ambient temperature, in Kelvin.
    P0 : float
        The total atmospheric pressure temperature, in Pa.
    T_e : float
        The exhaust temperature, in Kelvin.
    approx : boolean
        Approximate Kohler theory rather than include detailed calculation (default False)

    Returns
    -------
    EI : floats:
        The emission index of ice, in particles per kg of fuel burned. 

    """
    kappa = aerosol.kappa
    r_drys = aerosol.r_drys
    Nis = aerosol.Nis

    def es(T):
        """See :func:`pyrcel.thermo.es` for full documentation"""
        return 611.2 * np.exp(17.67 * T / (T + 243.5))

    pv_sat = es(T - 273.15)  # saturation vapor pressure
    e = (1.0 + Smax) * pv_sat  # water vapor pressure
    rho_air_dry = (P0 - e) / c.Rd / T # density of dry air, in kg/m^3

    ## Define parameters used to describe contrail dilution using one of two methods:

    ## (1) according to B Kärcher et al., 2015
    tau_m = 10e-3 # timescale over which contrail mixing is unaffected by entrainment, in seconds
    beta = 0.9 # plume dilution parameter

    ## (2) according to U Schumann et al., 1998 (used in Schuman et al., 2012)
    ## tau_m = 10 ** (np.log10((70/7000))/0.8)
    ## beta = 0.8
    
    expansion_time = tau_m * ((T_e - T_a) / (T - T_a)) ** (1/beta)
    dilution_parameter = np.where(expansion_time > tau_m, (tau_m/expansion_time) ** beta, 1) # dilution parameter as a function of expansion time

    if kappa == 0.500001:
        Nis = T_a/T * (1 - dilution_parameter) * aerosol.Nis
    else:
        Nis = dilution_parameter * (rho_air_dry)/((P0 * 28.96e-3) / (8.3145 * T_e)) * aerosol.Nis # dilution of number concentration

    dT_dt = - beta * (T_e - T_a)/tau_m * dilution_parameter ** (1 + 1/beta)

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
    
    ## Identify the frozen fraction for each of the bins
    ff = frozen_fraction(T, r_drys, rs, kappa, dT_dt)
    ff = ff.fillna(0.0) # replace nan values with 0.0
    ff = np.round(ff, decimals = 3)
    
    # Coerce the reference wet droplet sizes (rs) to an array if they were passed as a Series
    if hasattr(rs, "values"):
        rs = rs.values

    r_crits, s_crits = list(
        zip(*[kohler_crit(T, r_dry, kappa, False) for r_dry in r_drys])
    )
    s_crits = np.array(s_crits)
    r_crits = np.array(r_crits)

    ## Equilibrium calculation - all aerosol whose critical supersaturation > environmental supersaturation 
    aerosol_eq = (rs < r_crits) & (ff < 1) 

    ## Equilibrium water calculation - all aerosol whose critical supersaturation is less than the environmental supersaturation and whose T_hom < T (water droplets)
    activated_eq = (rs >= r_crits) & (ff < 1)

    ## Equilibrium ice calculation - all aerosol whose critical supersaturation is less than the environmental supersaturation and whose T_hom > T (ice crystals)
    frozen_eq = (ff >= 1) 

    ## Identify the bin radii and associated number concentration of non-activated aerosol particles, water droplets and ice crystals
    aerosol_mode_conc = Nis[aerosol_eq]
    aerosol_mode_radius = rs[aerosol_eq]

    ice_mode_conc = Nis[frozen_eq]
    ice_mode_radius = rs[frozen_eq]
    
    liquid_mode_conc = Nis[activated_eq]
    liquid_mode_radius = rs[activated_eq]

    ## Determine the effective emission index of ice
    EIi_eq = (np.sum(ice_mode_conc) * 60) / (dilution_parameter * rho_air_dry) # parameterization for emission index is taken from B Kärcher et al., 2015.
           
    return EIi_eq
    
# noinspection PyUnresolvedReferences
def multi_mode_activation(Smax, T, aerosols, rss):
    """Compute the activation statistics of a multi-mode, binned_activation
    aerosol population.

    Parameters
    ----------
    Smax : float
        Environmental maximum supersaturation.
    T : float
        Environmental temperature.
    aerosol : array of :class:`AerosolSpecies`
        The characterizations of the dry aerosols.
    rss : array of arrays of floats
        Wet radii corresponding to each aerosol/droplet population.

    Returns
    -------
    eqs, kns : lists of floats
        The activated fractions of each aerosol population.

    """
    act_fracs = []
    for rs, aerosol in zip(rss, aerosols):
        eq, kn, _, _ = binned_activation(Smax, T, rs, aerosol)
        act_fracs.append([eq, kn])
    return list(zip(*act_fracs))


######################################################################
# Code implementing the Nenes and Seinfeld (2003) parameterization,
# with improvements from Fountoukis and Nenes (2005), Barahona
# et al (2010), and Morales Betancourt and Nenes (2014)
######################################################################


def _vpres(T):
    """Polynomial approximation of saturated water vapour pressure as
    a function of temperature.

    Parameters
    ----------
    T : float
        Ambient temperature, in Kelvin

    Returns
    -------
    float
        Saturated water vapor pressure expressed in mb

    See Also
    --------
    es

    """
    # Coefficients for vapor pressure approximation
    A = [
        6.107799610e0,
        4.436518521e-1,
        1.428945805e-2,
        2.650648471e-4,
        3.031240396e-6,
        2.034080948e-8,
        6.136820929e-11,
    ]
    T -= 273  # Convert from Kelvin to C
    vp = A[-1] * T
    for ai in reversed(A[1:-1]):
        vp = (vp + ai) * T
    vp += A[0]
    return vp


def _erfp(x):
    """Polynomial approximation to error function"""
    AA = [0.278393, 0.230389, 0.000972, 0.078108]
    y = np.abs(1.0 * x)
    axx = 1.0 + y * (AA[0] + y * (AA[1] + y * (AA[2] + y * AA[3])))
    axx = axx * axx
    axx = axx * axx
    axx = 1.0 - (1.0 / axx)

    if x <= 0.0:
        axx = -1.0 * axx

    return axx


def mbn2014(
    V,
    T,
    P,
    aerosols=[],
    accom=c.ac,
    mus=[],
    sigmas=[],
    Ns=[],
    kappas=[],
    xmin=1e-5,
    xmax=0.1,
    tol=1e-6,
    max_iters=100,
):
    """Computes droplet activation using an iterative scheme.

    This method implements the iterative activation scheme under development by
    the Nenes' group at Georgia Tech. It encompasses modifications made over a
    sequence of several papers in the literature, culminating in [MBN2014]. The
    implementation here overrides some of the default physical constants and
    thermodynamic calculations to ensure consistency with a reference implementation.

    Parameters
    ----------
    V, T, P : floats
        Updraft speed (m/s), parcel temperature (K) and pressure (Pa)
    aerosols : list of :class:`AerosolSpecies`
        List of the aerosol population in the parcel; can be omitted if ``mus``,
        ``sigmas``, ``Ns``, and ``kappas`` are present. If both supplied, will
        use ``aerosols``.
    accom : float, optional (default=:const:`constants.ac`)
        Condensation/uptake accomodation coefficient
    mus, sigmas, Ns, kappas : lists of floats
        Lists of aerosol population parameters; must be present if ``aerosols``
        is not passed, but ``aerosols`` overrides if both are present
    xmin, xmax : floats, opional
        Minimum and maximum supersaturation for bisection
    tol : float, optional
        Convergence tolerance threshold for supersaturation, in decimal units
    max_iters : int, optional
        Maximum number of bisections before exiting convergence

    Returns
    -------
    smax, N_acts, act_fracs : lists of floats
        Maximum parcel supersaturation and the number concentration/activated
        fractions for each mode

    .. [MBN2014] Morales Betancourt, R. and Nenes, A.: Droplet activation
       parameterization: the population splitting concept revisited, Geosci.
       Model Dev. Discuss., 7, 2903-2932, doi:10.5194/gmdd-7-2903-2014, 2014.

    """

    # TODO: Convert mutable function arguments to None

    if aerosols:
        d = _unpack_aerosols(aerosols)
        mus = d["mus"]
        sigmas = d["sigmas"]
        kappas = d["kappas"]
        Ns = d["Ns"]
    else:
        # Assert that the aerosol was already decomposed into component vars
        assert mus is not None
        assert sigmas is not None
        assert Ns is not None
        assert kappas is not None

    # Convert sizes/number concentrations to diameters + SI units
    mus = np.asarray(mus)
    Ns = np.asarray(Ns)

    dpgs = 2 * (mus * 1e-6)
    Ns = Ns * 1e6

    nmodes = len(Ns)

    # Overriding using Nenes' physical constants, for numerical accuracy comparisons
    Ma = c.Ma  # 29e-3
    g = c.g  # 9.81
    Mw = c.Mw  # 18e-3
    R = c.R  # 8.31
    Rho_w = c.rho_w  # 1e3
    L = c.L  # 2.25e6
    Cp = c.Cp  # 1.0061e3

    # Thermodynamic environmental values to be set
    # TODO: Revert to functions in 'thermo' module
    # rho_a = rho_air(T, P, RH=0.0) # MBN2014: could set RH=1.0, account for moisture
    rho_a = P * Ma / R / T
    aka = ka_cont(T)  # MBN2014: could use thermo.ka(), include air density
    # surt = sigma_w(T)
    surt = 0.0761 - 1.55e-4 * (T - 273.0)

    # Compute modal critical supersaturation (Sg) for each mode, corresponding
    # to the critical supersaturation at the median diameter of each mode
    A = 4.0 * Mw * surt / R / T / Rho_w
    # There are three different ways to do this:
    #    1) original formula from MBN2014
    f = lambda T, dpg, kappa: np.sqrt((A**3.0) * 4.0 / 27.0 / kappa / (dpg**3.0))
    #    2) detailed kohler calculation
    # f = lambda T, dpg, kappa: kohler_crit(T, dpg/2., kappa)
    #    3) approximate kohler calculation
    # f = lambda T, dpg, kappa: kohler_crit(T, dpg/2., kappa, approx=True)
    # and the possibility of a correction factor:
    f2 = lambda T, dpg, kappa: np.exp(f(T, dpg, kappa)) - 1.0
    sgis = [f2(T, dpg, kappa) for dpg, kappa in zip(dpgs, kappas)]

    # Calculate the correction factor for the water vapor diffusivity.
    # Note that the Nenes' form is the exact same as the continuum form in this package
    # dv_orig = dv_cont(T, P)
    dv_orig = 1e-4 * (0.211 / (P / 1.013e5) * (T / 273.0) ** 1.94)
    dv_big = 5.0e-6
    dv_low = 1e-6 * 0.207683 * (accom ** (-0.33048))

    coef = (2.0 * np.pi * Mw / (R * T)) ** 0.5
    # TODO: Formatting on this average dv equation
    dv_ave = (dv_orig / (dv_big - dv_low)) * (
        (dv_big - dv_low)
        - (2 * dv_orig / accom)
        * coef
        * (
            np.log(
                (dv_big + (2 * dv_orig / accom) * coef)
                / (dv_low + (2 * dv_orig / accom) * coef)
            )
        )
    )

    # Setup constants used in supersaturation equation
    wv_pres_sat = _vpres(T) * (1e5 / 1e3)  # MBN2014: could also use thermo.es()
    alpha = g * Mw * L / Cp / R / T / T - g * Ma / R / T
    beta1 = P * Ma / wv_pres_sat / Mw + Mw * L * L / Cp / R / T / T
    beta2 = (
        R * T * Rho_w / wv_pres_sat / dv_ave / Mw / 4.0
        + L * Rho_w / 4.0 / aka / T * (L * Mw / R / T - 1.0)
    )  # this is 1/G
    beta = 0.5 * np.pi * beta1 * Rho_w / beta2 / alpha / V / rho_a

    cf1 = 0.5 * np.sqrt((1 / beta2) / (alpha * V))
    cf2 = A / 3.0

    def _sintegral(smax):
        """Integrate the activation equation, using ``spar`` as the population
        splitting threshold.

        Inherits the workspace thermodynamic/constant variables from one level
        of scope higher.
        """

        zeta_c = ((16.0 / 9.0) * alpha * V * beta2 * (A**2)) ** 0.25
        delta = 1.0 - (zeta_c / smax) ** 4.0  # spar -> smax
        critical = delta <= 0.0

        if critical:
            ratio = (
                (2e7 / 3.0) * A * (smax ** (-0.3824) - zeta_c ** (-0.3824))
            )  # Computing sp1 and sp2 (sp1 = sp2)
            ratio = 1.0 / np.sqrt(2.0) + ratio

            if ratio > 1.0:
                ratio = 1.0  # cap maximum value
            ssplt2 = smax * ratio

        else:
            ssplt1 = 0.5 * (1.0 - np.sqrt(delta))  # min root --> sp1
            ssplt2 = 0.5 * (1.0 + np.sqrt(delta))  # max root --> sp2
            ssplt1 = np.sqrt(ssplt1) * smax
            ssplt2 = np.sqrt(ssplt2) * smax

        ssplt = ssplt2  # secondary partitioning supersaturation

        # Computing the condensation integrals I1 and I2
        sum_integ1 = 0.0
        sum_integ2 = 0.0

        integ1 = np.empty(nmodes)
        integ2 = np.empty(nmodes)

        sqrtwo = np.sqrt(2.0)
        for i in range(nmodes):
            log_sigma = np.log(sigmas[i])  # ln(sigma_i)
            log_sgi_smax = np.log(sgis[i] / smax)  # ln(sg_i/smax)
            log_sgi_sp2 = np.log(sgis[i] / ssplt2)  # ln(sg_i/sp2

            u_sp2 = 2.0 * log_sgi_sp2 / (3.0 * sqrtwo * log_sigma)
            u_smax = 2.0 * log_sgi_smax / (3.0 * sqrtwo * log_sigma)
            # Subtract off the integrating factor
            log_factor = 3.0 * log_sigma / (2.0 * sqrtwo)

            d_eq = (
                A * 2.0 / sgis[i] / 3.0 / np.sqrt(3.0)
            )  # Dpc/sqrt(3) - equilibrium diameter

            erf_u_sp2 = _erfp(u_sp2 - log_factor)  # ERF2
            erf_u_smax = _erfp(u_smax - log_factor)  # ERF3

            integ2[i] = (
                np.exp(9.0 / 8.0 * log_sigma * log_sigma) * Ns[i] / sgis[i]
            ) * (erf_u_sp2 - erf_u_smax)

            if critical:
                u_sp_plus = sqrtwo * log_sgi_sp2 / 3.0 / log_sigma
                erf_u_sp_plus = _erfp(u_sp_plus - log_factor)

                integ1[i] = 0.0
                I_extra_term = (
                    Ns[i]
                    * d_eq
                    * np.exp((9.0 / 8.0) * log_sigma * log_sigma)
                    * (1.0 - erf_u_sp_plus)
                    * ((beta2 * alpha * V) ** 0.5)
                )  # 'inertially limited' particles

            else:
                g_i = np.exp((9.0 / 2.0) * log_sigma * log_sigma)
                log_sgi_sp1 = np.log(sgis[i] / ssplt1)  # ln(sg_i/sp1)

                int1_partial2 = Ns[i] * smax  # building I1(0, sp2), eq (B4)
                int1_partial2 *= (1.0 - _erfp(u_sp2)) - 0.5 * (
                    (sgis[i] / smax) ** 2.0
                ) * g_i * (1.0 - _erfp(u_sp2 + 3.0 * log_sigma / sqrtwo))

                u_sp1 = 2.0 * log_sgi_sp1 / (3.0 * sqrtwo * log_sigma)
                int1_partial1 = Ns[i] * smax  # building I1(0, sp1), eq (B4)
                int1_partial1 *= (1.0 - _erfp(u_sp1)) - 0.5 * (
                    (sgis[i] / smax) ** 2.0
                ) * g_i * (1.0 - _erfp(u_sp1 + 3.0 * log_sigma / sqrtwo))

                integ1[i] = int1_partial2 - int1_partial1  # I1(sp1, sp2)

                u_sp1_inertial = sqrtwo * log_sgi_sp1 / 3.0 / log_sigma
                erf_u_sp1 = _erfp(u_sp1_inertial - log_factor)
                I_extra_term = (
                    Ns[i]
                    * d_eq
                    * np.exp((9.0 / 8.0) * log_sigma * log_sigma)
                    * (1.0 - erf_u_sp1)
                    * ((beta2 * alpha * V) ** 0.5)
                )  # 'inertially limited' particles

            # Compute total integral values
            sum_integ1 += integ1[i] + I_extra_term
            sum_integ2 += integ2[i]

        return sum_integ1, sum_integ2

    # Bisection routine
    # Initial calculation
    x1 = xmin  # min cloud super sat -> 0.
    integ1, integ2 = _sintegral(x1)
    # Note that we ignore the contribution from the FHH integral term in this
    # implementation
    y1 = (integ1 * cf1 + integ2 * cf2) * beta * x1 - 1.0

    x2 = xmax  # max cloud super sat
    integ1, integ2 = _sintegral(x2)
    y2 = (integ1 * cf1 + integ2 * cf2) * beta * x2 - 1.0

    # Iteration of bisection routine to convergence
    iter_count = 0
    for i in range(max_iters):
        iter_count += 1

        x3 = 0.5 * (x1 + x2)
        integ1, integ2 = _sintegral(x3)
        y3 = (integ1 * cf1 + integ2 * cf2) * beta * x3 - 1.0

        if (y1 * y3) <= 0.0:  # different signs
            y2 = y3
            x2 = x3
        else:
            y1 = y3
            x1 = x3
        if np.abs(x2 - x1 <= tol * x1):
            break

    # Finalize bisection with one more intersection
    x3 = 0.5 * (x1 + x2)
    integ1, integ2 = _sintegral(x3)
    y3 = (integ1 * cf1 + integ2 * cf2) * beta * x3 - 1.0

    smax = x3

    n_acts, act_fracs = [], []
    for mu, sigma, N, kappa, sgi in zip(mus, sigmas, Ns, kappas, sgis):
        N_act, act_frac = lognormal_activation(
            smax, mu * 1e-6, sigma, N * 1e-6, kappa, sgi
        )
        n_acts.append(N_act)
        act_fracs.append(act_frac)

    return smax, n_acts, act_fracs


def arg2000(
    V,
    T,
    P,
    aerosols=[],
    accom=c.ac,
    mus=[],
    sigmas=[],
    Ns=[],
    kappas=[],
    min_smax=False,
):
    """Computes droplet activation using a psuedo-analytical scheme.

    This method implements the psuedo-analytical scheme of [ARG2000] to
    calculate droplet activation an an adiabatically ascending parcel. It
    includes the extension to multiple lognormal modes, and the correction
    for non-unity condensation coefficient [GHAN2011].

    To deal with multiple aerosol modes, the scheme includes an expression
    trained on the mode std deviations, :math:`\sigma_i`

    .. math::

        S_\\text{max} = 1 \\bigg/ \sqrt{\sum \\frac{1}{S^2_\text{mi}}\left[H(f_i, g_i)\right]}

    This effectively combines the supersaturation maximum for each mode into
    a single value representing competition between modes. An alternative approach,
    which assumes the mode which produces the smallest predict Smax sets a
    first-order control on the activation, is also available

    Parameters
    ----------
    V, T, P : floats
        Updraft speed (m/s), parcel temperature (K) and pressure (Pa)
    aerosols : list of :class:`AerosolSpecies`
        List of the aerosol population in the parcel; can be omitted if ``mus``,
        ``sigmas``, ``Ns``, and ``kappas`` are present. If both supplied, will
        use ``aerosols``.
    accom : float, optional (default=:const:`constants.ac`)
        Condensation/uptake accomodation coefficient
    mus, sigmas, Ns, kappas : lists of floats
        Lists of aerosol population parameters; must be present if ``aerosols``
        is not passed, but ``aerosols`` overrides if both are present.
    min_smax : boolean, optional
        If `True`, will use alternative formulation for parameterizing competition
        described above.

    Returns
    -------
    smax, N_acts, act_fracs : lists of floats
        Maximum parcel supersaturation and the number concentration/activated
        fractions for each mode

    .. [ARG2000] Abdul-Razzak, H., and S. J. Ghan (2000), A parameterization of
       aerosol activation: 2. Multiple aerosol types, J. Geophys. Res., 105(D5),
       6837-6844, doi:10.1029/1999JD901161.

    .. [GHAN2011] Ghan, S. J. et al (2011) Droplet Nucleation: Physically-based
       Parameterization and Comparative Evaluation, J. Adv. Model. Earth Syst.,
       3, doi:10.1029/2011MS000074

    """

    if aerosols:
        d = _unpack_aerosols(aerosols)
        mus = d["mus"]
        sigmas = d["sigmas"]
        kappas = d["kappas"]
        Ns = d["Ns"]
    else:
        # Assert that the aerosol was already decomposed into component vars
        assert mus is not None
        assert sigmas is not None
        assert Ns is not None
        assert kappas is not None

    # Originally from Abdul-Razzak 1998 w/ Ma. Need kappa formulation
    wv_sat = es(T - 273.15)
    alpha = (c.g * c.Mw * c.L) / (c.Cp * c.R * (T**2)) - (c.g * c.Ma) / (c.R * T)
    gamma = (c.R * T) / (wv_sat * c.Mw) + (c.Mw * (c.L**2)) / (c.Cp * c.Ma * T * P)

    # Condensation effects - base calculation
    G_a = (c.rho_w * c.R * T) / (wv_sat * dv_cont(T, P) * c.Mw)
    G_b = (c.L * c.rho_w * ((c.L * c.Mw / (c.R * T)) - 1)) / (ka_cont(T) * T)
    G_0 = 1.0 / (G_a + G_b)  # reference, no kinetic effects

    Smis = []
    Sparts = []
    for mu, sigma, N, kappa in zip(mus, sigmas, Ns, kappas):
        am = mu * 1e-6
        N = N * 1e6

        fi = 0.5 * np.exp(2.5 * (np.log(sigma) ** 2))
        gi = 1.0 + 0.25 * np.log(sigma)

        A = (2.0 * sigma_w(T) * c.Mw) / (c.rho_w * c.R * T)
        rc_mode, Smi2 = kohler_crit(T, am, kappa, approx=True)

        # Scale ``G`` to account for differences in condensation coefficient
        if accom == 1.0:
            G = G_0
        else:
            # Scale using the formula from [GHAN2011]
            # G_ac - estimate using critical radius of number mode radius,
            #        and new value for condensation coefficient
            G_a = (c.rho_w * c.R * T) / (wv_sat * dv(T, rc_mode, P, accom) * c.Mw)
            G_b = (c.L * c.rho_w * ((c.L * c.Mw / (c.R * T)) - 1)) / (ka_cont(T) * T)
            G_ac = 1.0 / (G_a + G_b)

            # G_ac1 - estimate using critical radius of number mode radius,
            #         unity condensation coefficient; re_use G_b (no change)
            G_a = (c.rho_w * c.R * T) / (wv_sat * dv(T, rc_mode, P, accom=1.0) * c.Mw)
            G_ac1 = 1.0 / (G_a + G_b)

            # Combine using scaling formula (40) from [GHAN2011]
            G = G_0 * G_ac / G_ac1

        # Parameterization integral solutions
        zeta = (2.0 / 3.0) * A * (np.sqrt(alpha * V / G))
        etai = ((alpha * V / G) ** (3.0 / 2.0)) / (N * gamma * c.rho_w * 2.0 * np.pi)

        # Contributions to maximum supersaturation
        Spa = fi * ((zeta / etai) ** (1.5))
        Spb = gi * (((Smi2**2) / (etai + 3.0 * zeta)) ** (0.75))
        S_part = (1.0 / (Smi2**2)) * (Spa + Spb)

        Smis.append(Smi2)
        Sparts.append(S_part)

    if min_smax:
        smax = 1e20
        for i in range(len(mus)):
            mode_smax = 1.0 / np.sqrt(Sparts[i])
            if mode_smax < smax:
                smax = mode_smax
    else:  # Use default competition parameterization
        smax = 1.0 / np.sqrt(np.sum(Sparts))

    n_acts, act_fracs = [], []
    for mu, sigma, N, kappa, sgi in zip(mus, sigmas, Ns, kappas, Smis):
        N_act, act_frac = lognormal_activation(smax, mu * 1e-6, sigma, N, kappa, sgi)
        n_acts.append(N_act)
        act_fracs.append(act_frac)

    return smax, n_acts, act_fracs


def shipwayabel2010(V, T, P, aerosol):
    """Activation scheme following Shipway and Abel, 2010
    (doi:10.1016/j.atmosres.2009.10.005).

    """
    raise NotImplementedError

    # rho_a = rho_air(T, P)
    #
    # # The following calculation for Dv_mean is identical to the Fountoukis and Nenes (2005)
    # # implementation, as referenced in Shipway and Abel, 2010
    # Dp_big = 5e-6
    # Dp_low = np.min([0.207683*(ac**-0.33048), 5.0])*1e-5
    # Dp_B = 2.*dv_cont(T, P)*np.sqrt(2*np.pi*Mw/R/T)/ac
    # Dp_diff = Dp_big - Dp_low
    # Dv_mean = (dv_cont(T, P)/Dp_diff)*(Dp_diff - Dp_B*np.log((Dp_big + Dp_B)/(Dp_low+Dp_B)))
    #
    # G = 1./rho_w/(Rv*T/es(T-273.15)/Dv_mean + (L/Ka/T)*(L/Rv/T - 1))
    #
    # # FROM APPENDIX B
    # psi1 = (g/T/Rd)*(Lv/Cp/T - 1.)
    # psi2 = (2.*np.pi*rho_w/rho_a) \
    #      * ((2.*G)**(3./2.))      \
    #      * (P/epsilon/es(T-273.15) + epsilon*(L**2)/Rd/(T**2)/Cp)
    #
    # Smax = 0
    #
    # act_fracs = []
    # #for Smi, aerosol in zip(Smis, aerosols):
    # #    ui = 2.*np.log(Smi/Smax)/(3.*np.sqrt(2.)*np.log(aerosol.distribution.sigma))
    # #    N_act = 0.5*aerosol.distribution.N*erfc(ui)
    # #    act_fracs.append(N_act/aerosol.distribution.N)
    #
    # return Smax, act_fracs


def ming2006(V, T, P, aerosol):
    """Ming activation scheme.

    NOTE - right now, the variable names correspond to the FORTRAN implementation of the routine. Will change in the future.

    """

    # TODO: rename variables
    # TODO: docstring
    # TODO: extend for multiple modes.

    raise NotImplementedError
    # Num = aerosol.Nis*1e-6
    #
    # RpDry = aerosol.distribution.mu*1e-6
    # kappa = aerosol.kappa
    #
    # # pre-algorithm
    # # subroutine Kohler()... calculate things from Kohler theory, particularly critical
    # # radii and supersaturations for each bin
    # r_crits, s_crits = list(zip(*[kohler_crit(T, r_dry, kappa) for r_dry in aerosol.r_drys]))
    #
    # # subroutine CalcAlphaGamma
    # alpha = (c.g*c.Mw*c.L)/(c.Cp*c.R*(T**2)) - (c.g*c.Ma)/(c.R*T)
    # gamma = (c.R*T)/(es(T-273.15)*c.Mw) + (c.Mw*(c.L**2))/(c.Cp*c.Ma*T*P)
    #
    # # re-name variables as in Ming scheme
    # Dpc = 2.*np.array(r_crits)*1e6
    # Dp0 = r_crits/np.sqrt(3.)
    # Sc = np.array(s_crits)+1.0
    # DryDp = aerosol.r_drys*2.
    #
    # # Begin algorithm
    # Smax1 = 1.0
    # Smax2 = 1.1
    #
    # iter_count = 1
    # while (Smax2 - Smax1) > 1e-7:
    #     #print "\t", iter_count, Smax1, Smax2
    #     Smax = 0.5*(Smax2 + Smax1)
    #     #print "---", Smax-1.0
    #
    #     ## subroutine Grow()
    #
    #     ## subroutine CalcG()
    #     # TODO: implement size-dependent effects on Dv, ka, using Dpc
    #     #G_a = (rho_w*R*T)/(es(T-273.15)*Dv_T(T)*Mw)
    #     G_a = (c.rho_w*c.R*T)/(es(T-273.15)*dv(T, (Dpc*1e-6)/2.)*c.Mw)
    #     #G_b = (L*rho_w*((L*Mw/(R*T))-1))/(ka_T(T)*T)
    #     G_b = (c.L*c.rho_w*((c.L*c.Mw/(c.R*T))-1))/(ka(T, 1.007e3, (Dpc*1e-6)/2.)*T)
    #     G = 1./(G_a + G_b) # multiply by four since we're doing diameter this time
    #
    #     Smax_large = (Smax > Sc) # if(Smax>Sc(count1,count2))
    #     WetDp = np.zeros_like(Dpc)
    #     #WetDp[Smax_large] = np.sqrt(Dpc[Smax_large]**2. + \
    #     #            1e12*(G[Smax_large]/(alpha*V))*((Smax-.0)**2.4 - (Sc[Smax_large]-.0)**2.4))
    #     WetDp[Smax_large] = 1e6*np.sqrt((Dpc[Smax_large]*1e-6)**2. + \
    #                         (G[Smax_large]/(alpha*V))*((Smax-.0)**2.4 - (Sc[Smax_large]-.0)**2.4))
    #
    #     #print Dpc
    #     #print WetDp/DryDp
    #     #print WetDp
    #
    #     # subroutine Activity()
    #     def Activity(dry, wet, dens, molar_weight):
    #         temp1 = (dry**3)*dens/molar_weight
    #         temp2 = ((wet**3) - (dry**3))*1e3/0.018
    #         act = temp2/(temp1+temp2)*np.exp(0.66/T/wet)
    #         #print dry[0], wet[0], dens, molar_weight, act[0]
    #         return act
    #     # Is this just the Kohler curve?
    #     Act = np.ones_like(WetDp)
    #     WetDp_large = (WetDp > 1e-5) # if(WetDp(i,j)>1e-5)
    #     Act[WetDp_large] = Seq(WetDp[WetDp_large]*1e-6, DryDp[WetDp_large], T, kappa) + 1.0
    #     #Act[WetDp_large] = Activity(DryDp[WetDp_large]*1e6, WetDp[WetDp_large], 1.7418e3, 0.132)
    #
    #     #print Act
    #
    #     # subroutine Conden()
    #
    #     # subroutine CalcG()
    #     # TODO: implement size-dependent effects on Dv, ka, using WetDp
    #     #G_a = (rho_w*R*T)/(es(T-273.15)*Dv_T(T)*Mw)
    #     G_a = (c.rho_w*c.R*T)/(es(T-273.15)*dv(T, (WetDp*1e-6)/2.)*c.Mw)
    #     #G_b = (L*rho_w*((L*Mw/(R*T))-1))/(ka_T(T)*T)
    #     G_b = (c.L*c.rho_w*((c.L*c.Mw/(c.R*T))-1))/(ka(T, 1.3e3, (WetDp*1e-6)/2.)*T)
    #     G = 1./(G_a + G_b) # multiply by four since we're doing diameter this time
    #
    #     WetDp_large = (WetDp > Dpc) # (WetDp(count1,count2)>Dpc(count1,count2))
    #     #WetDp_large = (WetDp > 0)
    #     f_stre = lambda x: "%12.12e" % x
    #     f_strf = lambda x: "%1.12f" % x
    #     #for i, a in enumerate(Act):
    #     #    if WetDp[i] > Dpc[i]:
    #     #        print "      ",i+1,  Act[i], f_stre(Smax-Act[i])
    #     CondenRate = np.sum((np.pi/2.)*1e3*G[WetDp_large]*(WetDp[WetDp_large]*1e-6)*Num[WetDp_large]*1e6*
    #                          (Smax-Act[WetDp_large]))
    #
    #     #print iter_count, "%r %r %r" % (Smax, CondenRate, alpha*V/gamma)
    #     DropletNum = np.sum(Num[WetDp_large])
    #     ActDp = 0.0
    #     for i in range(1, len(WetDp)):
    #         if (WetDp[i] > Dpc[i]) and (WetDp[i-1] < Dpc[i]):
    #             ActDp = DryDp[i]
    #
    #     # Iteration logic
    #     if CondenRate < (alpha*V/gamma):
    #         Smax1 = Smax*1.0
    #     else:
    #         Smax2 = Smax*1.0
    #
    #     iter_count += 1
    #
    # Smax = Smax-1.0
    #
    # return Smax, None
