"""
Physical Constants for Gravitational Wave Analysis
===================================================

All constants in SI units. Values sourced from:
- CODATA 2018 (via astropy.constants)
- IAU 2015 Resolution B3 (solar mass parameter)
- Particle Data Group 2022 (fundamental constants)

The gravitational wave community uses geometric units (G=c=1) internally
but we keep SI throughout and convert explicitly to avoid unit errors.
Every constant is documented with its source and uncertainty.

References
----------
[1] Mohr, Newell, Taylor, "CODATA 2018", Rev. Mod. Phys. 93, 025010 (2021)
[2] IAU 2015 Resolution B3: Nominal solar mass parameter GM_sun
[3] Maggiore, "Gravitational Waves: Theory and Experiments", Vol. 1 (2007)
"""

import numpy as np

# ---------------------------------------------------------------------------
# Fundamental constants (CODATA 2018 via astropy)
# ---------------------------------------------------------------------------
try:
    from astropy import constants as ac
    from astropy import units as u

    # Gravitational constant: G = 6.67430(15) × 10^-11 m^3 kg^-1 s^-2
    G_SI = ac.G.to(u.m**3 / (u.kg * u.s**2)).value

    # Speed of light: c = 299792458 m/s (exact by definition)
    C_SI = ac.c.to(u.m / u.s).value

    # Solar mass: M_sun = 1.98892 × 10^30 kg (from GM_sun / G)
    MSUN_SI = ac.M_sun.to(u.kg).value

    # Megaparsec: 1 Mpc = 3.0857 × 10^22 m
    MPC_SI = u.Mpc.to(u.m)

    # Planck constant (used in some QNM calculations)
    HBAR_SI = ac.hbar.to(u.J * u.s).value

    ASTROPY_AVAILABLE = True

except ImportError:
    # Fallback: hardcoded CODATA 2018 values if astropy is not available
    ASTROPY_AVAILABLE = False

    G_SI = 6.67430e-11       # m^3 kg^-1 s^-2, CODATA 2018
    C_SI = 299792458.0       # m/s, exact by definition
    MSUN_SI = 1.98892e30     # kg, IAU 2015 nominal
    MPC_SI = 3.085677581e22  # m, IAU 2012
    HBAR_SI = 1.054571817e-34  # J·s, CODATA 2018


# ---------------------------------------------------------------------------
# Derived constants used throughout the GW pipeline
# ---------------------------------------------------------------------------

# Gravitational radius of the Sun: R_g = G * M_sun / c^2
# This is the fundamental length scale in GW physics.
# For a 1 M_sun object, R_g ≈ 1.477 km
RSUN_SI = G_SI * MSUN_SI / C_SI**2  # meters, ≈ 1476.6 m

# Solar mass in seconds (geometric time unit): T_sun = G * M_sun / c^3
# This sets the timescale of GW emission.
# A 10 M_sun BH has a light-crossing time ~ 10 * T_sun ≈ 49 μs
TSUN_SI = G_SI * MSUN_SI / C_SI**3  # seconds, ≈ 4.926e-6 s

# Solar mass in inverse seconds (frequency unit): F_sun = c^3 / (G * M_sun)
# The characteristic GW frequency scales as 1/M, so
# f_char ~ F_sun / M_total (in solar masses)
FSUN_SI = C_SI**3 / (G_SI * MSUN_SI)  # Hz, ≈ 2.031e5 Hz

# Pi and related
PI = np.pi
TWOPI = 2.0 * np.pi
PI_SQ = np.pi**2

# Euler-Mascheroni constant — appears in tail contributions to the
# post-Newtonian gravitational wave flux at 1.5PN and higher orders.
# Reference: Blanchet, Living Reviews in Relativity 17, 2 (2014), Eq. (314)
EULER_GAMMA = 0.5772156649015329  # Euler-Mascheroni constant


# ---------------------------------------------------------------------------
# LIGO detector parameters
# ---------------------------------------------------------------------------

# Standard LIGO sample rate: 4096 Hz
# This is the native sample rate of h(t) strain data released by LIGO.
# Nyquist frequency = 2048 Hz, covering the full sensitive band.
LIGO_SAMPLE_RATE = 4096  # Hz

# Lower frequency cutoff for LIGO analysis
# Below ~20 Hz, seismic noise dominates and the detector is insensitive.
# Reference: LIGO-T1800044-v5, "Advanced LIGO sensitivity design curve"
LIGO_F_LOWER = 20.0  # Hz

# Audio output sample rate (CD quality)
AUDIO_SAMPLE_RATE = 44100  # Hz


# ---------------------------------------------------------------------------
# Utility conversion functions
# ---------------------------------------------------------------------------

def solar_masses_to_kg(m_solar):
    """Convert mass from solar masses to kilograms.

    Parameters
    ----------
    m_solar : float or array
        Mass in solar masses (M_sun).

    Returns
    -------
    float or array
        Mass in kilograms.
    """
    return m_solar * MSUN_SI


def kg_to_solar_masses(m_kg):
    """Convert mass from kilograms to solar masses.

    Parameters
    ----------
    m_kg : float or array
        Mass in kilograms.

    Returns
    -------
    float or array
        Mass in solar masses.
    """
    return m_kg / MSUN_SI


def mpc_to_meters(d_mpc):
    """Convert luminosity distance from Megaparsecs to meters.

    Parameters
    ----------
    d_mpc : float or array
        Distance in Megaparsecs.

    Returns
    -------
    float or array
        Distance in meters.
    """
    return d_mpc * MPC_SI


def meters_to_mpc(d_m):
    """Convert distance from meters to Megaparsecs.

    Parameters
    ----------
    d_m : float or array
        Distance in meters.

    Returns
    -------
    float or array
        Distance in Megaparsecs.
    """
    return d_m / MPC_SI


def chirp_mass(m1, m2):
    """Compute the chirp mass of a binary system.

    The chirp mass is the combination of component masses that determines
    the leading-order gravitational wave frequency evolution. It is the
    best-measured parameter from a GW inspiral signal.

    M_chirp = (m1 * m2)^(3/5) / (m1 + m2)^(1/5)

    Reference: Cutler & Flanagan, PRD 49, 2658 (1994), Eq. (2.5)

    Parameters
    ----------
    m1, m2 : float
        Component masses (any consistent unit system).

    Returns
    -------
    float
        Chirp mass in same units as input.
    """
    return (m1 * m2) ** (3.0 / 5.0) / (m1 + m2) ** (1.0 / 5.0)


def symmetric_mass_ratio(m1, m2):
    """Compute the symmetric mass ratio η of a binary system.

    η = m1 * m2 / (m1 + m2)^2

    η ranges from 0 (extreme mass ratio) to 1/4 (equal mass).
    It is the key parameter controlling post-Newtonian corrections
    beyond leading order.

    Reference: Blanchet, Living Reviews in Relativity 17, 2 (2014), Eq. (1)

    Parameters
    ----------
    m1, m2 : float
        Component masses (any consistent unit system).

    Returns
    -------
    float
        Symmetric mass ratio (dimensionless), 0 < η ≤ 0.25.
    """
    return m1 * m2 / (m1 + m2) ** 2


def reduced_mass(m1, m2):
    """Compute the reduced mass of a binary system.

    μ = m1 * m2 / (m1 + m2)

    Parameters
    ----------
    m1, m2 : float
        Component masses (any consistent unit system).

    Returns
    -------
    float
        Reduced mass in same units as input.
    """
    return m1 * m2 / (m1 + m2)


def effective_spin(m1, m2, s1z, s2z):
    """Compute the effective spin parameter χ_eff.

    χ_eff = (m1 * s1z + m2 * s2z) / (m1 + m2)

    This mass-weighted combination of aligned spin components is
    the dominant spin parameter in the GW phase evolution.
    It is conserved (approximately) throughout the inspiral.

    Reference: Ajith et al., PRL 106, 241101 (2011), Eq. (3)

    Parameters
    ----------
    m1, m2 : float
        Component masses (any consistent unit system).
    s1z, s2z : float
        Dimensionless spin components along orbital angular momentum,
        range [-1, 1] for Kerr black holes.

    Returns
    -------
    float
        Effective spin parameter (dimensionless), range [-1, 1].
    """
    return (m1 * s1z + m2 * s2z) / (m1 + m2)


def isco_frequency(m_total_solar):
    """Compute the gravitational wave frequency at the innermost
    stable circular orbit (ISCO) for a Schwarzschild (non-spinning) black hole.

    f_ISCO = c^3 / (6^(3/2) * π * G * M_total)

    This is the GW frequency (twice the orbital frequency) at which
    the inspiral phase formally ends and the plunge/merger begins.
    For a Schwarzschild BH, the ISCO radius is 6*G*M/c^2.

    Reference: Misner, Thorne & Wheeler, "Gravitation" (1973), §25.5
               Maggiore, "Gravitational Waves" Vol.1 (2007), Eq. (4.170)

    Parameters
    ----------
    m_total_solar : float
        Total mass of the binary in solar masses.

    Returns
    -------
    float
        ISCO GW frequency in Hz.
    """
    m_total_kg = m_total_solar * MSUN_SI
    return C_SI**3 / (6.0**(3.0 / 2.0) * PI * G_SI * m_total_kg)
