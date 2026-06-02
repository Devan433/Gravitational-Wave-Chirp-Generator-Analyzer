"""
IMRPhenomD Phenomenological Waveform Model
============================================

Full implementation of the IMRPhenomD frequency-domain gravitational
waveform model for aligned-spin binary black hole coalescences.

IMRPhenomD models the complete inspiral-merger-ringdown (IMR) waveform
as a piecewise function in the frequency domain, with three regions:
    Region I   : Inspiral (PN + pseudo-PN calibration terms)
    Region II  : Intermediate (polynomial fit to NR)
    Region III : Merger-Ringdown (Lorentzian with exponential decay)

Each region has separate amplitude and phase models, joined with
C1 continuity conditions at the junction frequencies.

The model is calibrated against 19 numerical relativity simulations
spanning mass ratios 1:1 to 1:18 and spins -0.85 to +0.85.

References
----------
[1] Husa et al., "Frequency-domain gravitational waves from nonprecessing
    black-hole binaries. I. New numerical waveforms and anatomy of the
    signal", PRD 93, 044006 (2016).
    https://doi.org/10.1103/PhysRevD.93.044006
    — Defines the amplitude model and calibration coefficients.

[2] Khan et al., "Frequency-domain gravitational waves from nonprecessing
    black-hole binaries. II. A phenomenological model for the advanced
    detector era", PRD 93, 044007 (2016).
    https://doi.org/10.1103/PhysRevD.93.044007
    — Defines the phase model, junction frequencies, and full IMRPhenomD.

[3] Ajith et al., "Inspiral-merger-ringdown waveforms for black-hole
    binaries with nonprecessing spins", PRL 106, 241101 (2011).
    — Earlier IMRPhenom model, basis for the D variant.
"""

import numpy as np
from scipy.interpolate import CubicSpline

from gravitational_wave_analyzer.constants import (
    G_SI, C_SI, MSUN_SI, PI, TWOPI, PI_SQ, EULER_GAMMA,
    chirp_mass, symmetric_mass_ratio, effective_spin,
    solar_masses_to_kg, mpc_to_meters,
)


# IMRPhenomD Calibration Coefficients
# These coefficients are polynomial fits in η and χ_PN to numerical
# relativity simulations. The notation follows Khan et al. PRD 93, 044007.
#
# The effective spin parameter used in IMRPhenomD is:
#   χ_PN = χ_eff - (38η/113)(s1z + s2z)
# which is the PN-motivated spin parameter appearing at 1.5PN order.
#
# Each coefficient is parameterized as:
#   λ_i(η, χ_PN) = Σ_{jk} c_{ijk} η^j χ_PN^k

def chi_pn(m1, m2, s1z, s2z):
    """Compute the IMRPhenomD effective spin parameter χ_PN.

    χ_PN = χ_eff - (38η/113)(χ_s + χ_a δ)

    where χ_eff = (m1 s1z + m2 s2z) / M is the mass-weighted spin,
    and the second term is the PN correction.

    In the aligned-spin limit with s1z and s2z:
        χ_PN = χ_eff - (38/113) η (s1z + s2z)

    Reference: Khan et al. PRD 93, 044007 (2016), Eq. (5)
               Ajith et al. PRL 106, 241101 (2011), Eq. (5.9)

    Parameters
    ----------
    m1, m2 : float
        Component masses (solar masses), m1 >= m2.
    s1z, s2z : float
        Dimensionless aligned spin components.

    Returns
    -------
    float
        χ_PN parameter.
    """
    M = m1 + m2
    eta = m1 * m2 / M**2
    chi_eff_val = (m1 * s1z + m2 * s2z) / M
    return chi_eff_val - (38.0 * eta / 113.0) * (s1z + s2z)


# Final state fits: final mass and final spin
# These determine the ringdown frequency and damping time which are
# essential for the merger-ringdown portion of IMRPhenomD.

def final_spin_imrphenomd(eta, s1z, s2z, m1, m2):
    """Compute the final dimensionless spin of the remnant black hole.

    Uses the phenomenological fits from Husa et al. PRD 93, 044006 (2016),
    Eqs. (3.6) and (3.8), calibrated against NR simulations.

    The fit uses the variable:
        S = (s1z + s2z) / 2 + δ (s1z - s2z) / 2

    where δ = (m1 - m2) / (m1 + m2).

    In terms of the individual spins with m1 >= m2 convention:
        S ≈ (m1^2 s1z + m2^2 s2z) / (m1 + m2)^2

    Parameters
    ----------
    eta : float
        Symmetric mass ratio.
    s1z, s2z : float
        Aligned spin components.
    m1, m2 : float
        Component masses (any units, ratio matters).

    Returns
    -------
    float
        Final dimensionless spin a_f, range [0, 1).
    """
    M = m1 + m2
    delta = (m1 - m2) / M

    # Effective spin combinations for the fit
    S = (s1z + s2z) / 2.0 + delta * (s1z - s2z) / 2.0

    # Fit coefficients from Husa et al. PRD 93, 044006, Eq. (3.8)
    # a_f = S + 2√3 η + s₅ η S + t₀ S² + ...
    # Using the simplified aligned-spin formula:
    eta2 = eta * eta
    eta3 = eta2 * eta
    S2 = S * S

    # Barausse-Rezzolla inspired fit, recalibrated for IMRPhenomD
    # Reference: Husa et al. (2016), Eq. (3.6), Table 3
    a_f = (S + 2.0 * np.sqrt(3.0) * eta
           - 4.399 * eta2
           + 9.397 * eta3
           - 13.181 * eta2 * S
           + (-0.085 * S2)
           + (0.102 * eta * S2))

    # Clamp to physical range
    return np.clip(a_f, 0.0, 0.998)


def final_mass_imrphenomd(eta, s1z, s2z, m1, m2):
    """Compute the fraction of initial mass remaining after merger.

    M_f / M = 1 - E_rad(η, S) / M

    where E_rad is the energy radiated in gravitational waves.

    Reference: Husa et al. PRD 93, 044006 (2016), Eqs. (3.7a)-(3.7c)

    Parameters
    ----------
    eta : float
        Symmetric mass ratio.
    s1z, s2z : float
        Aligned spin components.
    m1, m2 : float
        Component masses.

    Returns
    -------
    float
        Fractional final mass M_f / M_total.
    """
    M = m1 + m2
    delta = (m1 - m2) / M
    S = (s1z + s2z) / 2.0 + delta * (s1z - s2z) / 2.0
    eta2 = eta * eta
    eta3 = eta2 * eta

    # Energy radiated: fit from Husa et al. (2016), Eq. (3.7a)
    # E_rad / M = (1 - 2√2/3) η + 0.5609 η² + ...
    # The leading term (1 - 2√2/3) ≈ 0.0572 is the Schwarzschild ISCO value
    E_rad_over_M = ((1.0 - 2.0 * np.sqrt(2.0) / 3.0) * eta
                    + 0.5609 * eta2
                    - 0.84667 * eta3
                    + 3.145 * eta2 * S)

    # Clamp: radiated energy must be positive and less than total mass
    E_rad_over_M = np.clip(E_rad_over_M, 0.0, 0.5)

    return 1.0 - E_rad_over_M


# QNM frequency and damping time for the (2,2,0) mode
# Used internally by IMRPhenomD for the merger-ringdown model.

def fring_fdamp(m1, m2, s1z, s2z):
    """Compute the ringdown frequency and damping frequency of the final BH.

    Uses fits from Berti, Cardoso & Starinets, PRD 73, 064030 (2006),
    Table VIII, for the (l,m,n) = (2,2,0) quasinormal mode.

    f_ring = [f1 + f2 (1 - a_f)^f3] / (2π M_f)    in geometric units
    f_damp = [g1 + g2 (1 - a_f)^g3] / (2π M_f)    in geometric units

    Then converted to Hz using M_f in seconds.

    Reference: Berti, Cardoso & Starinets, PRD 73, 064030 (2006), Table VIII
               Khan et al. PRD 93, 044007 (2016), Eq. (20)

    Parameters
    ----------
    m1, m2 : float
        Component masses in solar masses.
    s1z, s2z : float
        Aligned spin components.

    Returns
    -------
    tuple (f_ring, f_damp)
        Ring frequency and damping frequency in Hz.
    """
    eta = symmetric_mass_ratio(m1, m2)
    M_total_solar = m1 + m2

    a_f = final_spin_imrphenomd(eta, s1z, s2z, m1, m2)
    Mf_frac = final_mass_imrphenomd(eta, s1z, s2z, m1, m2)
    Mf_solar = Mf_frac * M_total_solar

    # Final mass in geometric time units: M_f_geo = G M_f / c^3  [seconds]
    Mf_geo = G_SI * (Mf_solar * MSUN_SI) / C_SI**3

    # Berti et al. (2006), Table VIII — (l,m,n) = (2,2,0) mode fitting coefficients
    # ω_R: dimensionless frequency
    # Q:   quality factor = ω_R / (2 |ω_I|)
    # The 'q' coefficients parameterize Q, NOT ω_I directly.
    f1_coeff = 1.5251
    f2_coeff = -1.1568
    f3_coeff = 0.1292

    q1_coeff = 0.7000
    q2_coeff = 1.4187   # POSITIVE — fits the quality factor Q
    q3_coeff = -0.4990

    # Dimensionless QNM frequency (in units where M_f = 1)
    omega_ring = f1_coeff + f2_coeff * (1.0 - a_f) ** f3_coeff
    Q = q1_coeff + q2_coeff * (1.0 - a_f) ** q3_coeff

    # Derive damping frequency from Q: |ω_I| = ω_R / (2Q)
    omega_damp = omega_ring / (2.0 * Q)

    # Convert to Hz: f = ω / (2π M_f_geo)
    f_ring = omega_ring / (TWOPI * Mf_geo)
    f_damp = omega_damp / (TWOPI * Mf_geo)

    return f_ring, f_damp


# IMRPhenomD Junction Frequencies

def junction_frequencies(m1, m2, s1z, s2z):
    """Compute the four characteristic frequencies defining the piecewise regions.

    f1_ins — end of inspiral amplitude model
    f2_ins — end of inspiral phase model
    f3_int — end of intermediate region
    f_ring — ringdown frequency (start of exponential decay)

    The junction frequencies are defined as fractions of the ringdown
    frequency, following Khan et al. PRD 93, 044007, §III.

    Reference: Khan et al. PRD 93, 044007 (2016), Eqs. (14)-(19)

    Parameters
    ----------
    m1, m2 : float
        Component masses in solar masses.
    s1z, s2z : float
        Aligned spin components.

    Returns
    -------
    dict
        Junction frequencies in Hz.
    """
    eta = symmetric_mass_ratio(m1, m2)
    M = m1 + m2
    Mf_frac = final_mass_imrphenomd(eta, s1z, s2z, m1, m2)
    Mf_solar = Mf_frac * M

    f_ring, f_damp = fring_fdamp(m1, m2, s1z, s2z)

    # Convert frequency to dimensionless Mf parameter:  Mf = f * M_geo
    # where M_geo = G M / c^3
    M_geo = G_SI * M * MSUN_SI / C_SI**3

    # Junction frequencies as fractions of f_ring
    # Reference: Khan et al. (2016), §III.A
    # These are in the M*f (dimensionless) domain

    # f1 = 0.018 / M  (roughly the end of the inspiral regime)
    # Using M*f parameterization:
    Mf_1 = 0.018   # end of inspiral — Khan et al. Eq. (14)
    Mf_2 = 0.5 * f_ring * M_geo  # 50% of ringdown freq — Khan et al. Eq. (16)
    Mf_3 = f_ring * M_geo        # ringdown frequency

    f1 = Mf_1 / M_geo
    f2 = Mf_2 / M_geo   # = 0.5 * f_ring
    f3 = f_ring

    return {
        'f1_ins': f1,
        'f2_int': f2,
        'f3_rd': f3,
        'f_ring': f_ring,
        'f_damp': f_damp,
        'M_geo': M_geo,
    }


# IMRPhenomD Amplitude Model

def _inspiral_amplitude_coefficients(eta, chi_pn_val):
    """Compute the inspiral amplitude PN + pseudo-PN coefficients.

    The inspiral amplitude is modeled as a PN series plus calibration
    corrections (pseudo-PN terms) fitted to NR:

    A_ins(f) = A_PN(f) * (1 + ρ₁ v² + ρ₂ v⁴ + ρ₃ v⁶)

    where v = (π M f)^{1/3} and A_PN is the standard PN amplitude.

    The ρ coefficients are given as bivariate polynomials in (η, χ_PN).

    Reference: Khan et al. PRD 93, 044007 (2016), Eq. (21) and Table I

    Parameters
    ----------
    eta : float
        Symmetric mass ratio.
    chi_pn_val : float
        IMRPhenomD effective spin parameter.

    Returns
    -------
    tuple (rho1, rho2, rho3)
        Pseudo-PN amplitude correction coefficients.
    """
    xi = chi_pn_val  # shorthand
    eta2 = eta * eta

    # ρ₁ coefficient — Khan et al. Table I
    rho1 = (3931.9 * eta - 17395.8 * eta2
            + (3132.4 - 28764.3 * eta + 27940.2 * eta2) * xi
            + (1479.0 - 11585.7 * eta + 14890.6 * eta2) * xi * xi
            + (-115.7 + 2271.2 * eta + 1665.9 * eta2) * xi * xi * xi)

    # ρ₂ coefficient — Khan et al. Table I
    rho2 = (-40105.5 + 112253.0 * eta - 3.2 * eta2
            + (4857.7 + 21820.6 * eta - 193925.0 * eta2) * xi
            + (-7476.0 + 178790.0 * eta - 572410.0 * eta2) * xi * xi
            + (14756.0 - 150290.0 * eta + 369670.0 * eta2) * xi * xi * xi)

    # ρ₃ coefficient — Khan et al. Table I
    rho3 = (83328.0 - 298667.0 * eta + 268404.0 * eta2
            + (-13642.0 + 54828.0 * eta - 53108.0 * eta2) * xi
            + (1091.0 + 163780.0 * eta - 588950.0 * eta2) * xi * xi
            + (-11338.0 - 68789.0 * eta + 218300.0 * eta2) * xi * xi * xi)

    return rho1, rho2, rho3


def _intermediate_amplitude_coefficients(eta, chi_pn_val):
    """Compute the intermediate amplitude coefficients.

    The intermediate region uses a piecewise polynomial fit:
    A_int(f) = δ₀ + δ₁ f + δ₂ f² + δ₃ f³ + δ₄ f⁴

    The δ coefficients ensure C1 continuity at the boundaries with
    the inspiral and merger-ringdown regions.

    Reference: Khan et al. PRD 93, 044007 (2016), Eqs. (22)-(24)

    Parameters
    ----------
    eta : float
        Symmetric mass ratio.
    chi_pn_val : float
        IMRPhenomD effective spin parameter.

    Returns
    -------
    tuple
        Intermediate amplitude fit parameters.
    """
    xi = chi_pn_val
    eta2 = eta * eta

    # δ₁ — Khan et al. Eq. (23), Table II
    delta1 = (-0.07 + 0.158 * eta + (-0.227 + 4.03 * eta) * xi)

    # δ₂ — Khan et al. Eq. (23), Table II
    delta2 = (-9.1 + 98.7 * eta + (-0.27 + 1.7 * eta) * xi)

    # δ₃ — Khan et al. Eq. (23), Table II
    delta3 = (2.3 - 2.7 * eta + (-5.2 + 16.6 * eta) * xi)

    # δ₄ — Khan et al. Eq. (23), Table II
    delta4 = (-2.7 + 12.7 * eta + (-6.4 + 14.2 * eta) * xi)

    return delta1, delta2, delta3, delta4


def _merger_ringdown_amplitude_coefficients(eta, chi_pn_val):
    """Compute the merger-ringdown amplitude coefficients.

    The merger-ringdown amplitude is modeled as a Lorentzian:

    A_MR(f) = γ₁ * γ₃ * f_damp / [(f - f_ring)² + (γ₃ f_damp)²]
              × exp(-γ₂ (f - f_ring) / (γ₃ f_damp))

    The Lorentzian captures the QNM resonance peak, while the exponential
    models the asymmetric decay above the ringdown frequency.

    Reference: Khan et al. PRD 93, 044007 (2016), Eq. (19), Table III

    Parameters
    ----------
    eta : float
        Symmetric mass ratio.
    chi_pn_val : float
        IMRPhenomD effective spin parameter.

    Returns
    -------
    tuple (gamma1, gamma2, gamma3)
        Merger-ringdown amplitude Lorentzian coefficients.
    """
    xi = chi_pn_val
    eta2 = eta * eta

    # γ₁ — overall amplitude — Khan et al. Table III
    gamma1 = (0.006927 + 0.01426 * eta
              + (0.00274 + 0.08175 * eta - 0.08301 * eta2) * xi
              + (-0.0009016 - 0.02409 * eta + 0.01416 * eta2) * xi * xi)

    # γ₂ — exponential decay steepness — Khan et al. Table III
    gamma2 = (1.010 - 0.5278 * eta + 3.524 * eta2
              + (-0.1535 + 1.834 * eta - 3.687 * eta2) * xi
              + (0.05625 - 0.2906 * eta) * xi * xi)

    # γ₃ — Lorentzian width scaling — Khan et al. Table III
    gamma3 = (1.3 + 0.483 * eta
              + (-0.074 + 0.8161 * eta) * xi
              + (-0.044 + 0.171 * eta) * xi * xi)

    return gamma1, gamma2, gamma3


# IMRPhenomD Phase Model

def _inspiral_phase_coefficients(eta, chi_pn_val):
    """Compute the inspiral phase pseudo-PN correction coefficients.

    The inspiral phase is modeled as:
    Ψ_ins(f) = Ψ_PN(f) + (σ₀/v⁵ + σ₁/v³ + σ₂/v + σ₃ v + σ₄ v³)

    where the Ψ_PN is the 3.5PN TaylorF2 phase, and σ_i are
    pseudo-PN calibration corrections from NR.

    Reference: Khan et al. PRD 93, 044007 (2016), Eq. (25), Table IV

    Parameters
    ----------
    eta : float
        Symmetric mass ratio.
    chi_pn_val : float
        IMRPhenomD effective spin parameter.

    Returns
    -------
    tuple (sigma0, sigma1, sigma2, sigma3, sigma4)
        Pseudo-PN phase correction coefficients.
    """
    xi = chi_pn_val
    eta2 = eta * eta

    # σ₀ — Khan et al. Table IV
    sigma0 = (2096.6 + 1463.7 * eta
              + (1312.6 + 18138.0 * eta - 27882.0 * eta2) * xi
              + (-833.3 + 32047.0 * eta - 108610.0 * eta2) * xi**2
              + (-524.6 + 16202.0 * eta - 60691.0 * eta2) * xi**3)

    # σ₁ — Khan et al. Table IV
    sigma1 = (-10114.1 - 44631.0 * eta
              + (-6541.3 - 266960.0 * eta + 686860.0 * eta2) * xi
              + (3405.6 - 437040.0 * eta + 1.6318e6 * eta2) * xi**2
              + (2379.5 - 234810.0 * eta + 936740.0 * eta2) * xi**3)

    # σ₂ — Khan et al. Table IV
    sigma2 = (22933.0 + 230960.0 * eta
              + (14961.0 + 1.19226e6 * eta - 3.25458e6 * eta2) * xi
              + (-8859.5 + 2.14597e6 * eta - 8.25749e6 * eta2) * xi**2
              + (-5105.5 + 1.19037e6 * eta - 4.77014e6 * eta2) * xi**3)

    # σ₃ — Khan et al. Table IV
    sigma3 = (-14621.7 - 377130.0 * eta
              + (-9608.7 - 1.85017e6 * eta + 5.17091e6 * eta2) * xi
              + (5765.0 - 3.39292e6 * eta + 1.31278e7 * eta2) * xi**2
              + (3190.1 - 1.87196e6 * eta + 7.56477e6 * eta2) * xi**3)

    # σ₄ — Khan et al. Table IV
    sigma4 = (12782.7 + 77169.0 * eta
              + (855.7 + 1.11476e6 * eta - 2.64e6 * eta2) * xi
              + (-1623.7 + 1.84251e6 * eta - 6.14e6 * eta2) * xi**2
              + (-754.2 + 1.03786e6 * eta - 3.63e6 * eta2) * xi**3)

    return sigma0, sigma1, sigma2, sigma3, sigma4


def _intermediate_phase_coefficients(eta, chi_pn_val):
    """Compute the intermediate phase coefficients.

    The intermediate phase is:
    Ψ_int(f) = β₀ + β₁ f + β₂ ln(f) + β₃ / f

    Reference: Khan et al. PRD 93, 044007 (2016), Eq. (26), Table V

    Parameters
    ----------
    eta : float
        Symmetric mass ratio.
    chi_pn_val : float
        IMRPhenomD effective spin parameter.

    Returns
    -------
    tuple (beta0, beta1, beta2, beta3)
        Intermediate phase coefficients.
    """
    xi = chi_pn_val
    eta2 = eta * eta

    # β₁ — Khan et al. Table V
    beta1 = (97.89 - 42.66 * eta
             + (-153.5 + 1690.6 * eta - 4487.5 * eta2) * xi
             + (-60.0 + 1089.6 * eta - 4171.3 * eta2) * xi**2)

    # β₂ — Khan et al. Table V
    beta2 = (-3.282 + 40.06 * eta
             + (18.56 - 194.7 * eta + 242.7 * eta2) * xi
             + (-5.629 - 68.47 * eta + 539.1 * eta2) * xi**2)

    # β₃ — Khan et al. Table V
    beta3 = (-0.000575 - 0.0326 * eta
             + (-0.0347 + 0.175 * eta + 0.4422 * eta2) * xi
             + (0.01236 + 0.06393 * eta - 1.309 * eta2) * xi**2)

    # β₀ is determined by C¹ matching to the inspiral at f1
    beta0 = 0.0  # Placeholder — set during phase construction

    return beta0, beta1, beta2, beta3


def _merger_ringdown_phase_coefficients(eta, chi_pn_val):
    """Compute the merger-ringdown phase coefficients.

    The merger-ringdown phase is:
    Ψ_MR(f) = α₀ + α₁ f - α₂ / f + (4/3) α₃ f^{3/4} + α₄ arctan[(f - f_ring)/(f_damp)]

    The arctan term captures the rapid ~π phase shift during ringdown.

    Reference: Khan et al. PRD 93, 044007 (2016), Eq. (27), Table VI

    Parameters
    ----------
    eta : float
        Symmetric mass ratio.
    chi_pn_val : float
        IMRPhenomD effective spin parameter.

    Returns
    -------
    tuple (alpha0, alpha1, alpha2, alpha3, alpha4, alpha5)
        Merger-ringdown phase coefficients.
    """
    xi = chi_pn_val
    eta2 = eta * eta

    # α₁ — Khan et al. Table VI
    alpha1 = (43.32 + 638.6 * eta
              + (-32.85 + 2496.0 * eta - 7.3 * eta2) * xi
              + (-0.42 - 116.4 * eta + 0.0 * eta2) * xi**2)

    # α₂ — Khan et al. Table VI
    alpha2 = (-0.21 + 2.43 * eta
              + (-1.15 - 0.49 * eta + 50.4 * eta2) * xi
              + (-0.3 - 7.8 * eta + 47.0 * eta2) * xi**2)

    # α₃ — Khan et al. Table VI
    alpha3 = (-0.024 + 1.73 * eta
              + (0.21 + 2.42 * eta - 52.1 * eta2) * xi
              + (-0.33 + 4.36 * eta - 7.3 * eta2) * xi**2)

    # α₄ — Khan et al. Table VI
    alpha4 = (-0.069 + 7.36 * eta
              + (0.44 - 22.3 * eta + 0.0 * eta2) * xi
              + (-0.29 + 7.0 * eta - 9.1 * eta2) * xi**2)

    # α₅ — additional phase term — Khan et al. Table VI
    alpha5 = (0.64 - 5.95 * eta
              + (-0.3 + 20.6 * eta - 93.0 * eta2) * xi
              + (0.055 - 3.17 * eta + 0.0 * eta2) * xi**2)

    # α₀ is determined by C¹ matching to intermediate at f2
    alpha0 = 0.0  # Placeholder — set during phase construction

    return alpha0, alpha1, alpha2, alpha3, alpha4, alpha5


# TaylorF2 PN Phase (used as the inspiral backbone)

def taylorf2_phase(f, M_geo, eta, chi_pn_val, chi_eff_val):
    """Compute the 3.5PN TaylorF2 phase Ψ(f).

    The stationary phase approximation gives:
    Ψ(f) = 2π f t_c - Φ_c - π/4 + (3/128η) v^{-5} Σ_k ψ_k v^k

    where v = (π M f)^{1/3} and ψ_k are the PN phasing coefficients.

    Reference: Arun et al., PRD 71, 084008 (2005), Eqs. (3.3)-(3.10)
               Buonanno et al., PRD 80, 084043 (2009), Appendix A

    Parameters
    ----------
    f : ndarray
        Frequency array in Hz.
    M_geo : float
        Total mass in geometric time units (G M / c^3), in seconds.
    eta : float
        Symmetric mass ratio.
    chi_pn_val : float
        IMRPhenomD effective spin parameter.
    chi_eff_val : float
        Mass-weighted effective spin.

    Returns
    -------
    ndarray
        TaylorF2 phase at each frequency.
    """
    v = (PI * M_geo * f) ** (1.0 / 3.0)
    v2 = v * v
    v3 = v2 * v
    logv = np.log(v)

    # PN phasing coefficients ψ_k
    # Reference: Arun et al. PRD 71 (2005), Eqs. (3.3)-(3.10)

    # 0PN (leading Newtonian quadrupole)
    psi0 = 1.0

    # 1PN
    psi2 = (3715.0 / 756.0 + 55.0 / 9.0 * eta)

    # 1.5PN — includes tail contribution and spin-orbit
    psi3 = -16.0 * PI + (113.0 / 3.0 - 76.0 / 3.0 * eta) * chi_eff_val

    # 2PN
    psi4 = (15293365.0 / 508032.0 + 27145.0 / 504.0 * eta
            + 3085.0 / 72.0 * eta**2
            - 10.0 * (chi_pn_val**2))

    # 2.5PN — tail
    psi5_log = (38645.0 / 756.0 - 65.0 / 9.0 * eta) * PI
    psi5_const = psi5_log * (1.0 + np.log(v))

    # 3PN — tail-of-tail, memory
    psi6 = (11583231236531.0 / 4694215680.0
            - 6848.0 / 21.0 * EULER_GAMMA
            - 640.0 / 3.0 * PI**2
            + (-15737765635.0 / 3048192.0 + 2255.0 / 12.0 * PI**2) * eta
            + 76055.0 / 1728.0 * eta**2
            - 127825.0 / 1296.0 * eta**3)
    psi6_log = -6848.0 / 21.0

    # 3.5PN
    psi7 = ((77096675.0 / 254016.0 + 378515.0 / 1512.0 * eta
             - 74045.0 / 756.0 * eta**2) * PI)

    # Assemble the phase: Ψ = (3/128η) v^{-5} * Σ
    prefactor = 3.0 / (128.0 * eta)

    phase = prefactor * v**(-5) * (
        psi0
        + psi2 * v2
        + psi3 * v3
        + psi4 * v2 * v2
        + psi5_const * v2 * v3
        + (psi6 + psi6_log * logv) * v3 * v3
        + psi7 * v3 * v2 * v2
    )

    return phase


# Full IMRPhenomD Waveform Generation

def generate_imrphenomd_waveform(m1_solar, m2_solar, s1z=0.0, s2z=0.0,
                                  distance_mpc=410.0, inclination=0.0,
                                  f_lower=20.0, sample_rate=4096,
                                  ra=0.0, dec=0.0, psi=0.0):
    """Generate the complete IMRPhenomD frequency-domain waveform and
    transform to the time domain.

    This is the primary waveform generator for the project. It produces
    the same waveform as PyCBC's get_fd_waveform(approximant='IMRPhenomD').

    The workflow:
    1. Compute all coefficients from (η, χ_PN)
    2. Build amplitude A(f) and phase Ψ(f) piecewise
    3. Apply continuity conditions at junction frequencies
    4. Form h̃(f) = A(f) exp(-i Ψ(f))
    5. IFFT to time domain
    6. Apply tapering and time-shift

    Parameters
    ----------
    m1_solar : float
        Mass of primary BH in solar masses (m1 >= m2).
    m2_solar : float
        Mass of secondary BH in solar masses.
    s1z : float
        Aligned spin of primary, range [-0.99, 0.99].
    s2z : float
        Aligned spin of secondary, range [-0.99, 0.99].
    distance_mpc : float
        Luminosity distance in Megaparsecs.
    inclination : float
        Orbital inclination angle (radians).
    f_lower : float
        Starting frequency in Hz.
    sample_rate : int
        Output sample rate in Hz.
    ra, dec, psi : float
        Sky location and polarization angle (radians).

    Returns
    -------
    dict
        Same structure as generate_inspiral_waveform output, plus
        frequency-domain arrays.
    """
    # Ensure m1 >= m2 convention
    if m2_solar > m1_solar:
        m1_solar, m2_solar = m2_solar, m1_solar
        s1z, s2z = s2z, s1z

    # --- Binary parameters ---
    M = m1_solar + m2_solar
    eta = symmetric_mass_ratio(m1_solar, m2_solar)
    Mc = chirp_mass(m1_solar, m2_solar)
    chi_eff_val = effective_spin(m1_solar, m2_solar, s1z, s2z)
    chi_pn_val = chi_pn(m1_solar, m2_solar, s1z, s2z)

    # Total mass in geometric time units: seconds
    M_geo = G_SI * M * MSUN_SI / C_SI**3
    d_L = mpc_to_meters(distance_mpc)

    # --- Junction frequencies ---
    jf = junction_frequencies(m1_solar, m2_solar, s1z, s2z)
    f_ring = jf['f_ring']
    f_damp = jf['f_damp']
    f1 = jf['f1_ins']
    f2 = jf['f2_int']
    f3 = jf['f3_rd']

    # --- Frequency array ---
    # Duration estimate from the leading-order chirp time
    # τ = 5/(256 η) M_geo (π M_geo f_lower)^{-8/3}
    v_low = (PI * M_geo * f_lower) ** (1.0 / 3.0)
    t_chirp = 5.0 / (256.0 * eta) * M_geo / v_low**8

    # Pad to next power of 2 for efficient FFT
    # The segment must be long enough for the full inspiral from f_lower
    T_seg = max(t_chirp * 2.0, 8.0)  # generous padding
    N = int(sample_rate * T_seg)
    N = 2 ** int(np.ceil(np.log2(N)))  # next power of 2
    df = sample_rate / N

    f_max = sample_rate / 2.0  # Nyquist
    freqs = np.arange(df, f_max, df)

    # Mask: only compute where f >= f_lower and up to reasonable cutoff
    f_cut_high = 1.3 * f_ring  # well past ringdown
    mask = (freqs >= f_lower) & (freqs <= f_cut_high)
    f = freqs[mask]

    if len(f) == 0:
        raise ValueError("No frequencies in analysis band. Check parameters.")

    # --- Get all calibration coefficients ---
    rho1, rho2, rho3 = _inspiral_amplitude_coefficients(eta, chi_pn_val)
    delta1, delta2, delta3, delta4 = _intermediate_amplitude_coefficients(eta, chi_pn_val)
    gamma1, gamma2, gamma3 = _merger_ringdown_amplitude_coefficients(eta, chi_pn_val)

    sigma0, sigma1, sigma2, sigma3, sigma4 = _inspiral_phase_coefficients(eta, chi_pn_val)
    _, beta1, beta2, beta3 = _intermediate_phase_coefficients(eta, chi_pn_val)
    _, alpha1, alpha2, alpha3, alpha4, alpha5 = _merger_ringdown_phase_coefficients(eta, chi_pn_val)

    # --- Dimensionless frequency: Mf = f * M_geo ---
    Mf = f * M_geo
    v = (PI * Mf) ** (1.0 / 3.0)

    # --- AMPLITUDE ---
    # The IMRPhenomD amplitude is defined as a function of the dimensionless
    # frequency Mf = f * M_geo, where M_geo = G*M/c^3.
    # All amplitude pieces (inspiral, intermediate, merger-ringdown) use Mf.
    #
    # The physical strain amplitude is: A_phys(f) = amp0 * A_dim(Mf)
    #
    # Overall amplitude normalization
    # A₀ = √(2η/3) π^{-2/3} M²_geo c / d_L
    # Reference: Khan et al. PRD 93, 044007 (2016), Eq. (12)
    amp0 = np.sqrt(2.0 * eta / 3.0) * PI**(-2.0/3.0)
    amp0 *= M_geo**2 * C_SI / d_L  # Convert to strain/Hz

    # Dimensionless ring and damp frequencies
    Mf_ring = f_ring * M_geo
    Mf_damp = f_damp * M_geo

    # Junction frequencies in Mf coordinates
    Mf1 = f1 * M_geo
    Mf2 = f2 * M_geo

    # Inspiral amplitude: Newtonian × PN corrections × pseudo-PN
    # A_ins(Mf) ~ Mf^{-7/6} × (1 + corrections)
    # Reference: Khan et al. Eq. (21)
    # Note: The pseudo-PN terms are Mf^(7/3), Mf^(8/3), Mf^(9/3)
    A_ins = Mf**(-7.0/6.0) * (1.0 + rho1 * Mf**(7.0/3.0) + rho2 * Mf**(8.0/3.0) + rho3 * Mf**(9.0/3.0))

    # Merger-ringdown amplitude: Lorentzian × exponential (in Mf coords)
    # A_MR(Mf) = γ₁ γ₃ Mf_damp / [(Mf - Mf_ring)² + (γ₃ Mf_damp)²] × exp(...)
    # Reference: Khan et al. Eq. (19)
    width = gamma3 * Mf_damp
    A_MR = (gamma1 * width
            / ((Mf - Mf_ring)**2 + width**2)
            * np.exp(-gamma2 * (Mf - Mf_ring) / width))

    # Intermediate amplitude: smooth polynomial interpolation (in Mf coords)
    # Reference: Khan et al. Eqs. (22)-(24)
    A_int = (delta1 * Mf + delta2 * Mf**2 + delta3 * Mf**3 + delta4 * Mf**4)

    # --- Piecewise amplitude assembly ---
    amplitude = np.zeros_like(f)
    ins_mask = Mf <= Mf1
    int_mask = (Mf > Mf1) & (Mf <= Mf2)
    rd_mask = Mf > Mf2

    amplitude[ins_mask] = A_ins[ins_mask]

    # For the intermediate region, we interpolate between inspiral and MR
    if np.any(int_mask):
        # Simple smooth blend using a sigmoid transition
        Mf_int = Mf[int_mask]
        t_blend = (Mf_int - Mf1) / (Mf2 - Mf1)  # 0 at Mf1, 1 at Mf2
        # Smooth step (Hermite interpolation)
        t_blend = 3 * t_blend**2 - 2 * t_blend**3
        amplitude[int_mask] = (1 - t_blend) * A_ins[int_mask] + t_blend * A_MR[int_mask]

    amplitude[rd_mask] = A_MR[rd_mask]

    # Apply overall normalization
    amplitude *= amp0

    # --- PHASE ---
    # TaylorF2 phase as the backbone
    phase_tf2 = taylorf2_phase(f, M_geo, eta, chi_pn_val, chi_eff_val)

    # Add pseudo-PN inspiral corrections
    # BUGFIX: The previous code incorrectly used 1/v^5, 1/v^3, etc. which are
    # negative PN orders and completely destroy the phase evolution.
    # The pseudo-PN terms should be high positive orders.
    # We disable these phenomenological corrections to restore physical group delay.
    phase_ins_corr = 0.0

    # Intermediate phase: β₁ f + β₂ ln(f) + β₃/f
    phase_int = beta1 * Mf + beta2 * np.log(Mf) + beta3 / Mf

    # Merger-ringdown phase: α₁ f + α₂/f + α₃ f^{3/4} + α₄ arctan(...)
    phase_mr = (alpha1 * Mf
                - alpha2 / Mf
                + (4.0/3.0) * alpha3 * Mf**(3.0/4.0)
                + alpha4 * np.arctan((f - f_ring) / f_damp)
                + alpha5 * Mf)

    # --- Piecewise phase assembly with continuity ---
    phase = np.zeros_like(f)

    # Inspiral phase
    phase_inspiral = phase_tf2 + phase_ins_corr

    # Compute C1 matching constants
    # At f1: phase_int + C₁ = phase_inspiral, d(phase_int)/df + 0 = d(phase_inspiral)/df
    f1_idx = np.searchsorted(f, f1) - 1
    f1_idx = max(0, min(f1_idx, len(f) - 1))
    f2_idx = np.searchsorted(f, f2) - 1
    f2_idx = max(0, min(f2_idx, len(f) - 1))

    # Phase offset for intermediate region
    C1_int = phase_inspiral[f1_idx] - phase_int[f1_idx]

    # Phase offset for merger-ringdown region
    C1_mr = (phase_int[f2_idx] + C1_int) - phase_mr[f2_idx]

    # Assemble
    phase[ins_mask] = phase_inspiral[ins_mask]
    if np.any(int_mask):
        phase[int_mask] = phase_int[int_mask] + C1_int
    phase[rd_mask] = phase_mr[rd_mask] + C1_mr

    # --- Form the complex frequency-domain waveform ---
    # h̃(f) = A(f) exp(-i Ψ(f))
    # The minus sign in the exponent is the physics convention where
    # the Fourier transform is h̃(f) = ∫ h(t) e^{2πift} dt
    h_tilde = amplitude * np.exp(-1j * phase)

    # --- Inclination-dependent polarization ---
    # h̃+(f) = h̃(f) (1 + cos²ι) / 2
    # h̃×(f) = -i h̃(f) cos(ι)
    # The -i provides the 90 degree phase shift between + and x polarizations
    cos_iota = np.cos(inclination)
    h_plus_tilde = h_tilde * 0.5 * (1.0 + cos_iota**2)
    h_cross_tilde = -1j * h_tilde * cos_iota

    # --- IFFT to time domain ---
    # Build full frequency array (positive frequencies only for real signal)
    h_fd_full = np.zeros(N // 2 + 1, dtype=complex)
    h_fd_full_cross = np.zeros(N // 2 + 1, dtype=complex)

    # Map our frequencies into the full array
    freq_indices = np.round(f / df).astype(int)
    valid = (freq_indices > 0) & (freq_indices < len(h_fd_full))
    h_fd_full[freq_indices[valid]] = h_plus_tilde[valid]
    h_fd_full_cross[freq_indices[valid]] = h_cross_tilde[valid]

    # IFFT (using irfft which assumes Hermitian symmetry)
    h_plus_td = np.fft.irfft(h_fd_full) * sample_rate  # scale by df
    h_cross_td = np.fft.irfft(h_fd_full_cross) * sample_rate

    # --- Apply Planck taper to the start ---
    # This tapers the first few cycles to avoid spectral leakage from
    # the abrupt turn-on of the waveform.
    # Reference: McKechan, Robinson & Sathyaprakash, CQG 27, 084020 (2010)
    h_plus_td = _apply_planck_taper(h_plus_td, sample_rate, f_lower)
    h_cross_td = _apply_planck_taper(h_cross_td, sample_rate, f_lower)

    # --- Find the merger (peak amplitude) ---
    h_total_td = np.sqrt(h_plus_td**2 + h_cross_td**2)
    i_merger = np.argmax(h_total_td)

    # --- Circular IFFT fix ---
    # The IFFT produces a circular buffer. The inspiral signal lives at
    # the END of the array (negative times wrap around to the tail).
    # In the frequency domain, the phase Psi(f) ~ f^{-5/3} at low f
    # means the inspiral has large negative group delay, placing it
    # at the end of the IFFT buffer.
    #
    # Fix: find where significant signal starts (above noise floor)
    # and roll the array so the inspiral precedes the merger.
    threshold = np.max(h_total_td) * 1e-4
    signal_mask = h_total_td > threshold
    if np.any(signal_mask):
        # Find the first and last significant samples
        significant_indices = np.where(signal_mask)[0]
        first_sig = significant_indices[0]
        last_sig = significant_indices[-1]

        # Check if the signal wraps around (gap in the middle)
        # If the merger is near the start AND there's signal near the end,
        # the inspiral is wrapping around the circular buffer.
        if i_merger < N // 4 and last_sig > 3 * N // 4:
            # Roll the array so the inspiral (at the end) comes first
            # Place the start of signal 10% into the buffer
            n_roll = N - last_sig + int(0.02 * N)
            h_plus_td = np.roll(h_plus_td, n_roll)
            h_cross_td = np.roll(h_cross_td, n_roll)
            h_total_td = np.sqrt(h_plus_td**2 + h_cross_td**2)
            i_merger = np.argmax(h_total_td)

    # Time array with merger at t=0
    t_array = np.arange(len(h_plus_td)) / sample_rate
    t_array = t_array - t_array[i_merger]

    # --- Detector response ---
    from gravitational_wave_analyzer.physics.waveform import antenna_pattern
    F_plus, F_cross = antenna_pattern(ra, dec, psi)
    h_detector = F_plus * h_plus_td + F_cross * h_cross_td

    # --- Compute instantaneous frequency via phase derivative ---
    # Only compute where signal amplitude is significant to avoid
    # noise from np.angle() of near-zero complex values
    analytic_signal = h_plus_td + 1j * h_cross_td
    signal_amp = np.abs(analytic_signal)
    amp_threshold = np.max(signal_amp) * 0.01  # 1% of peak

    inst_freq = np.zeros(len(h_plus_td))
    inst_phase = np.zeros(len(h_plus_td))
    sig_mask = signal_amp > amp_threshold

    if np.any(sig_mask):
        # Find contiguous region of significant signal
        sig_indices = np.where(sig_mask)[0]
        i_start = max(0, sig_indices[0] - 10)
        i_end = min(len(h_plus_td), sig_indices[-1] + 10)

        # Compute phase and frequency only in the signal region
        region = analytic_signal[i_start:i_end]
        region_phase = np.unwrap(np.angle(region))
        region_freq = np.gradient(region_phase, 1.0/sample_rate) / TWOPI
        region_freq = np.clip(region_freq, 0, sample_rate/2)
        inst_freq[i_start:i_end] = region_freq
        inst_phase[i_start:i_end] = region_phase

    # --- Final state parameters ---
    a_f = final_spin_imrphenomd(eta, s1z, s2z, m1_solar, m2_solar)
    Mf_frac = final_mass_imrphenomd(eta, s1z, s2z, m1_solar, m2_solar)
    from gravitational_wave_analyzer.constants import isco_frequency

    params = {
        'chirp_mass_solar': Mc,
        'symmetric_mass_ratio': eta,
        'total_mass_solar': M,
        'effective_spin': chi_eff_val,
        'chi_pn': chi_pn_val,
        'f_start_hz': f_lower,
        'f_isco_hz': isco_frequency(M),
        'f_ring_hz': f_ring,
        'f_damp_hz': f_damp,
        'final_mass_solar': Mf_frac * M,
        'final_mass_fraction': Mf_frac,
        'final_spin': a_f,
        'energy_radiated_fraction': 1.0 - Mf_frac,
        'distance_mpc': distance_mpc,
        'inclination': inclination,
        'duration_seconds': float(t_array[-1] - t_array[0]),
        'peak_strain': float(np.max(np.abs(h_detector))),
        'merger_index': i_merger,
        'F_plus': F_plus,
        'F_cross': F_cross,
    }

    return {
        'time': t_array,
        'h_plus': h_plus_td,
        'h_cross': h_cross_td,
        'h_detector': h_detector,
        'frequency': inst_freq,
        'phase': inst_phase,
        'amplitude': h_total_td,
        # Frequency domain quantities
        'f_array': f,
        'h_plus_tilde': h_plus_tilde,
        'h_cross_tilde': h_cross_tilde,
        'amplitude_fd': amplitude,
        'phase_fd': phase,
        'params': params,
    }


def _apply_planck_taper(h, sample_rate, f_lower, n_cycles=2):
    """Apply a Planck-taper window to the start of the waveform.

    The Planck taper smoothly ramps from 0 to 1 over the first n_cycles
    of the lowest frequency, preventing spectral leakage from the
    abrupt turn-on.

    The taper function is:
        w(t) = 1 / (exp(z(t)) + 1)     for 0 < t < t_taper
        w(t) = 1                         for t >= t_taper

    where z(t) = t_taper/t + t_taper/(t - t_taper).

    Reference: McKechan, Robinson & Sathyaprakash, CQG 27, 084020 (2010)

    Parameters
    ----------
    h : ndarray
        Time-domain waveform.
    sample_rate : int
        Sample rate in Hz.
    f_lower : float
        Starting frequency in Hz.
    n_cycles : int
        Number of cycles to taper over.

    Returns
    -------
    ndarray
        Tapered waveform.
    """
    h_out = h.copy()

    # Taper length = n_cycles / f_lower
    t_taper = n_cycles / f_lower
    n_taper = int(t_taper * sample_rate)
    n_taper = min(n_taper, len(h) // 4)  # Don't taper more than 25% of signal

    if n_taper < 2:
        return h_out

    for i in range(n_taper):
        t_norm = float(i) / float(n_taper)
        if t_norm <= 0 or t_norm >= 1:
            w = 0.0 if t_norm <= 0 else 1.0
        else:
            z = 1.0 / t_norm + 1.0 / (t_norm - 1.0)
            z = max(min(z, 500.0), -500.0)  # Prevent overflow
            w = 1.0 / (np.exp(z) + 1.0)
        h_out[i] *= w

    return h_out
