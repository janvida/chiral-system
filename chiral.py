# -*- coding: utf-8 -*-
'''so that 1/2 == 0.5, and not 0'''
from __future__ import division
from matplotlib import pyplot as plt
import tinyarray
import numpy as np
from math import pi, sin, cos

import kwant

# define Pauli-matrices for convenience
sigma_0 = tinyarray.array([[1, 0], [0, 1]])
sigma_x = tinyarray.array([[0, 1], [1, 0]])
sigma_y = tinyarray.array([[0, -1j], [1j, 0]])
sigma_z = tinyarray.array([[1, 0], [0, -1]])


def plot_bandstructure(flead, momenta, show=False, J="N/A", spin="N/A"):
    """Compute the band structure for each transverse
    subband.
    """
    bands = kwant.physics.Bands(flead)
    energies = [bands(k) for k in momenta]
    if show is True:
        plt.figure()
        plt.title("Band Structure (J=" + str(J) + ", Spin=" + spin + ")")
        plt.plot(momenta, energies)
        plt.xlabel("Momentum [a^-1]")
        plt.ylabel("Energy [t]")
        plt.grid(True)
        plt.show()
    return energies


def density_of_states(sys, energies, L, W, show=False, J="N/A", spin="N/A"):
    """Calculate the density of states"""
    dos = []
    for energy in energies:
        dos.append(np.sum(kwant.ldos(sys, energy)))
    dos = [x / (W*L) for x in dos]
    if show is True:
        plt.figure()
        plt.title("Density of States (J=" + str(J) + ", Spin=" + spin + ")")
        plt.plot(energies, dos)
        plt.xlabel("Energies [t]")
        plt.ylabel("Density of States [Energy^-1 m^-2]")
        plt.grid(True)
        plt.show()
    return dos


def s_matrix(sys, energies):
    """Compute the S-Matrix as a function of energy.
    Return a list of s of type
    <class 'kwant.solvers.common.SMatrix'>.
    """
    s = []
    for energy in energies:
        s.append(kwant.smatrix(sys, energy))
    return s


def conductance(smatrix, energies, to_lead, from_lead, show=False,
                J="N/A", spin="N/A"):
    """Calculate and plot conductance. Return conductance."""
    cond = []
    for i in range(len(smatrix)):
        cond.append(smatrix[i].transmission(to_lead, from_lead))
    if show is True:
        plt.figure()
        plt.plot(energies, cond)
        plt.xlabel("Energy [t]")
        plt.ylabel("Conductance [e^2/h]")
        plt.title("Conductance from Left to Right (J=" + str(J) + ", Spin=" + spin + ")")
        plt.grid(True)
        plt.show()
    return cond


def quant_axis(theta, phi):
    if theta == 0 and phi == 0:
        spin = 'Sz'
    elif theta != 0 and phi == 0:
        spin = 'Sx'
    elif theta != 0 and phi != 0:
        spin = 'Sy'
    return spin


def plot_wave_function(sys, n):
    """Calculates and plots the wavefunction of order n.
    i.e. The ground state is n = 0, 1 and the first excited
    state is n = 2, 3 (due to spin degeneracy).
    """
    import scipy.linalg as la
    ham = sys.hamiltonian_submatrix()
    evecs = la.eigh(ham)[1]
    # print evecs.shape
    wf_up = abs(evecs[::2, n])**2
    wf_down = abs(evecs[1::2, n])**2
    wf = wf_up + wf_down
    kwant.plotter.map(sys, wf_up, oversampling=1, cmap='gist_heat_r')
    kwant.plotter.map(sys, wf_down, oversampling=1, cmap='gist_heat_r')
    kwant.plotter.map(sys, wf, oversampling=1, cmap='gist_heat_r')


def plot_system(sys, L, W, dx, dy, chiral):
    if chiral == 0:
        dx = 0
        dy = 0

    def site_color(site):
        if site.pos in helix(L, W, dx, dy):
            return 'magenta'
        else:
            return 'black'

    def site_size(site):
        if site.pos in helix(L, W, dx, dy):
            return 0.3
        else:
            return 0.2

    kwant.plot(sys, site_color=site_color, site_size=site_size,
               colorbar=None)


def helix(L, W, dx, dy):
    """This function returns a list of the points.
    in the scattering LxW which fall onto the helix with
    slope (pitch) dx, dy.
    """
    if dx == 0 and dy == 0:
        return []
    else:
        points = []
        for i in range(0, int(L/dx)):
            __ = [i*dx, (i*dy) % W]
            points.append(__)
        return points

def spin_conductance(sys, energies, lead_out, lead_in, sigma=sigma_z):
    """Calculate the spin conductance between two leads.
    Uses the expression
        G =  Tr[ σ_{α} Γ_{q} G_{qp} Γ_{p} G^+_{qp} ]   (1)
    Where  Γ_{q} is the coupling matrix to lead q ( = i[Σ - Σ^+] )
    and G_{qp} is the submatrix of the system's Greens function
    between sites interfacing with lead p and q (not to be confused
    with the "G" on the left-hand-side, which is the spin conductance
    between the two leads).
    Parameters
    ----------
    G : `kwant.solvers.common.GreensFunction`
        The Greens function of the system as returned by
        `kwant.greens_function`.
    lead_out : positive integer
        The lead where spin current is collected
    lead_in : positive integer
        The lead where spin current is injected
    sigma : `numpy.ndarray` of shape (2, 2)
        The Pauli matrix of the quantization axis along
        which to measure the spin current
    Notes
    -----
    Assumes that the spin structure is encoded in the matrix structure
    of the Hamiltonian elements (i.e. there are not separate lattices/
    sites for the spin degree of freedom). If you have the spin degree
    of freedom on a separate lattice already you can trivially get
    the spin conductance by using separate spin up/down leads.
    See http://dx.doi.org/10.1103/PhysRevB.89.195418 for a derivation
    of equation (1).
    """
    # calculate Γ G Γ G^+
    energy = energies
    G = kwant.greens_function(sys, energy)
    ttdagger = G._a_ttdagger_a_inv(lead_out, lead_in)
    shp = attdagger.shape[0] // sigma.shape[0]
    # build spin matrix over whole lead interface
    sigma_matrix = np.kron(np.eye(shp), sigma)
    return np.trace(sigma_matrix.dot(ttdagger)).real


def make_system(a=1, t=1.0, L=30, W=10, dx=0, dy=0, chiral=0, shift=0,
                rashba=0, lead_shift=0, J=0, theta=0, phi=0, E0_left=0):
    """Create a tight-binding system on a square lattice.

    Keyword arguments:
    a -- Lattice constant (default 1)
    t -- Hopping amplitude (default 1)
    L -- Length of scattering region (default 30)
    W -- Width of scattering region (default 10)
    dx -- Horizontal component of helix pitch (default 0)
    dy -- Vertical component of helix pitch (default 0)
    chiral -- Strength of onsite helical potential (default 0)
    """
    lat = kwant.lattice.square(a)
    sys = kwant.Builder()

    # Define the onsite potential
    def onsite(site):
        if site.pos in helix(L, W, dx, dy):
            return 4 * t * sigma_0 + chiral * sigma_0
        else:
            return 4 * t * sigma_0

    # Define the scattering region.
    sys[(lat(x, y) for x in range(L) for y in range(W))] = (
        onsite)
    # Hoppings in the x-direction
    sys[kwant.builder.HoppingKind((1, 0), lat, lat)] = (
        -t * sigma_0 - 1j * rashba * sigma_y)
    # Hoppings in y-directions
    sys[kwant.builder.HoppingKind((0, 1), lat, lat)] = (
        -t * sigma_0 + 1j * rashba * sigma_x)
    # Hoppings for period boundary conditions
    for x in range(0, L-shift):
        sys[lat(x, 0), lat(x+shift, W-1)] = (-t * sigma_0
                                             + 1j * rashba * sigma_x)

    # Define left lead
    left_lead = kwant.Builder(kwant.TranslationalSymmetry(
        (-a, 0)))
    left_lead[(lat(0, j) for j in range(W))] = (4 * t * sigma_0
                                                + E0_left * sigma_0
                                                + J *
                                                (sin(theta)*cos(phi) *
                                                    sigma_x
                                                 + sin(theta)*sin(phi) *
                                                    sigma_y
                                                 + cos(theta)*sigma_z))
    left_lead[kwant.builder.HoppingKind((1, 0), lat, lat)] = (
        -t * sigma_0)
    left_lead[kwant.builder.HoppingKind((0, 1), lat, lat)] = (
        -t * sigma_0)
    # Periodic boundary conditions in the lead
    left_lead[lat(1, 0), lat(1+lead_shift, W-1)] = (-t * sigma_0)
    sys.attach_lead(left_lead)

    # Define right lead
    right_lead = kwant.Builder(kwant.TranslationalSymmetry(
        (a, 0)))
    right_lead[(lat(0, j) for j in range(W))] = (
        4 * t * sigma_0)
    right_lead[kwant.builder.HoppingKind((1, 0), lat, lat)] = (
        -t * sigma_0)
    right_lead[kwant.builder.HoppingKind((0, 1), lat, lat)] = (
        -t * sigma_0)
    # Periodic boundary conditions in the lead
    right_lead[lat(1, 0), lat(1+lead_shift, W-1)] = (-t * sigma_0)
    sys.attach_lead(right_lead)

    return sys, left_lead, right_lead


def main():
    L = 30
    W = 10
    dx = 1
    dy = 3
    chiral = 0
    rashba = 0.03
    shift = 1
    # Right now, this only works for lead_shift = 1.
    # How to fix this??...how to change period in lead?
    lead_shift = 1
    J = 2
    theta = pi/2
    phi = pi/2
    spin = quant_axis(theta, phi)
    E0_left = 0

    for j in [1, -1]:
        J = J * j

        sys, left_lead, right_lead = make_system(L=L, W=W, dx=dx, dy=dy,
                                                 chiral=chiral,
                                                 shift=shift,
                                                 rashba=rashba,
                                                 lead_shift=lead_shift,
                                                 J=J, theta=theta,
                                                 phi=phi,
                                                 E0_left=E0_left)

        plot_system(sys, L, W, dx, dy, chiral)

        sys = sys.finalized()
        left_lead = left_lead.finalized()
        right_lead = right_lead.finalized()

        momenta = [-pi + 0.02*pi*i for i in range(100)]
        plot_bandstructure(left_lead, momenta, show=True, J=J, spin=spin)

        energies = [0.1*i for i in range(81)]
        s = s_matrix(sys, energies)
        g_rl = conductance(s, energies, 1, 0, show=True, J=J, spin=spin)
        density_of_states(sys, energies, L, W, show=True, J=J, spin=spin)

        # print spin_conductance(sys, energies[2], 1, 0)

# Call the main function if the script gets executed (as
# opposed to imported).
# See <http://docs.python.org/library/__main__.html>.
if __name__ == '__main__':
    main()
