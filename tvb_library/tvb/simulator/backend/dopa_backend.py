# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Scientific Package. This package holds all simulators, and
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package. See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2024, Baycrest Centre for Geriatric Care ("Baycrest") and others
#
# This program is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software Foundation,
# either version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE.  See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this
# program.  If not, see <http://www.gnu.org/licenses/>.
#
#
#   CITATION:
# When using The Virtual Brain for scientific publications, please cite it as explained here:
# https://www.thevirtualbrain.org/tvb/zwei/neuroscience-publications
#
#

"""
Numba backend which uses templating to generate simulation
code.

.. moduleauthor:: Marmaduke Woodman <marmaduke.woodman@univ-amu.fr>

"""

import os
import importlib
import numpy as np
import autopep8


from tvb.simulator.lab import *
from numba import jit


class DOPABackend(object):
    def check_compatibility(self, sim): 
        def check_choices(val, choices):
            if not isinstance(val, choices):
                raise NotImplementedError("Unsupported simulator component. Given: {}\nExpected one of: {}".format(val, choices))
        # monitors
        if len(sim.monitors) > 1:
            raise NotImplementedError("Configure with one monitor.")
        check_choices(sim.monitors[0], (monitors.Raw, monitors.TemporalAverage))
        # integrators
        check_choices(sim.integrator, integrators.HeunStochastic)
        # models
        check_choices(sim.model, models.MPRDopa)
        # coupling
        check_choices(sim.coupling, coupling.Linear)
        # surface
        if sim.surface is not None:
            raise NotImplementedError("Surface simulation not supported.")
        # stimulus evaluated outside the backend, no restrictions


    def run_sim(self, sim, conn_i, conn_d, g_i, g_d, nstep=None, simulation_length=None, chunksize=100000, compatibility_mode=False, print_source=False):
        assert nstep is not None or simulation_length is not None or sim.simulation_length is not None

        self.check_compatibility(sim)
        if nstep is None:
            if simulation_length is None:
                simulation_length = sim.simulation_length
            nstep = int(np.ceil(simulation_length/sim.integrator.dt))
                
        if isinstance(sim.monitors[0], monitors.Raw):
            noise = sim.integrator.noise.generate((sim.model.nvar, sim.connectivity.number_of_regions, nstep))
            noise_gfun = sim.integrator.noise.gfun(0)
            dt = sim.integrator.dt
            noise *= noise_gfun * np.sqrt(dt)
            pars = tuple(self._get_par(sim.model, attr_name)[0] for attr_name in sim.model.parameter_names)
            svar_bufs, c = run_sim_plain(dopa_dfun, pars, sim.initial_conditions, noise, dt, sim.connectivity.weights, conn_i, conn_d, sim.coupling.a, g_i, g_d, nstep, compatibility_mode=compatibility_mode, print_source=print_source)
            time = np.arange(svar_bufs[0].shape[1]) * sim.integrator.dt
        # elif isinstance(sim.monitors[0], monitors.TemporalAverage):
        #     svar_bufs = self._run_sim_tavg_chunked(sim, nstep, chunksize=chunksize, compatibility_mode=compatibility_mode, print_source=print_source)
        #     T = sim.monitors[0].period
        #     time = np.arange(svar_bufs[0].shape[1]) * T + 0.5 * T
        else:
            raise NotImplementedError("Only Raw or TemporalAverage monitors supported.")
        return (time, svar_bufs), c

    def _get_par(self, obj, foostring):
        return getattr(obj, foostring)

    def _time_average(self, ts, istep):
        N, T = ts.shape
        return np.mean(ts.reshape(N,T//istep,istep),-1) # length of ts better be multiple of istep 

@jit
def dopa_dfun(X, coupling, pars):
    c_inh, c_exc, c_dopa = coupling  # This zero refers to the second element of cvar (V in this case)
    a,b,c,ga,gg,eta,Delta,I,Ea,Eg,Sja,Sjg,tauSa,tauSg,alpha,beta,ud,k,Vmax,Km,Bd,Ad,tauDp = pars
    # not change a, b, c, Sja, Sjg, tauSa, tauSg, Vmax
    r, V, u, Sa, Sg, Dp = X[0,:], X[1,:], X[2,:], X[3,:], X[4,:], X[5,:]
    derivative = np.stack((2. * a * r * V + b * r - (Ad * Dp + Bd)* ga * Sa * r - gg * Sg * r + (a * Delta) / np.pi,
    a * V**2 + b * V + c + eta - (np.pi**2 * r**2) / a + (Ad * Dp + Bd) * ga * Sa * (Ea - V) + gg * Sg * (Eg - V) + I - u,
    alpha * (beta * V - u) + ud * r,
    -Sa / tauSa + Sja * (r + c_exc),
    -Sg / tauSg + Sjg * c_inh,
    (k * c_dopa - Vmax * Dp / (Km + Dp)) / tauDp))
    return derivative


@jit
def run_sim_plain(dfun, pars, X0, dW, dt, conn_e, conn_i, conn_d, g_e, g_i, g_d, nstep=None, compatibility_mode=False, print_source=True):
    connectivities = np.stack((conn_i, conn_e, conn_d))
    X = X0[0,:,:,0]
    y_all = np.empty((nstep//10,)+X.shape)
    t_all = np.empty((nstep//10, ))

    t_all[0] = 0
    y_all[0, :] = X
    count=0
    t = 0
    i = 0
    couplings = []
    for step in np.arange(nstep):
        dw = dW[:,:,step]
        coupling = cx(X, connectivities, g_i, g_e, g_d)
        couplings.append(coupling)
        if np.any(np.isnan(coupling[1])):
            print("warning, coupling is nan, stopping sim")
            print(step)
            break
        m_dx_tn = dfun(X, coupling, pars)
        m_dx_tn = mpr_dopa_positive(m_dx_tn)
        inter = X + dt * m_dx_tn + dw
        dX = (m_dx_tn + dfun(inter, coupling, pars)) * dt / 2.0
        X = X + dX + dw
        X = mpr_dopa_positive(X)
        t += dt
        if  (count % 10)==0 and (i< (t_all.shape[0]-1)):
            i+=1
            t_all[i]=t
            y_all[i,:]= X
    return y_all, couplings


@jit
def cx(state_vars, connectivities, g_i, g_e, g_d):
    r = state_vars[0,:]
    aff_inhibitor = connectivities[0,...].T @ r * g_i
    aff_excitator = connectivities[1,...].T @ r * g_e
    aff_dopamine = connectivities[2,...].T @ r * g_d
    return np.stack((aff_inhibitor, aff_excitator, aff_dopamine))

@jit
def mpr_dopa_positive(X):
    r, V, u, Sa, Sg, Dp = X[0,:], X[1,:], X[2,:], X[3,:], X[4,:], X[5,:]
    return np.concatenate((r*(r>0), V, u, Sa*(Sa>0), Sg*(Sg>0), Dp*(Dp>0))).reshape(X.shape)