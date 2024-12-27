from tvb.simulator.lab import *
import numpy as np
from tvb.simulator.backend.dopa_backend import DOPABackend
import matplotlib.pyplot as plt
plt.ion()

def configure_conn(conn_file, conn_speed):
    conn = connectivity.Connectivity.from_file( conn_file )
    conn.speed = np.array([conn_speed])
    np.fill_diagonal(conn.weights, 0.)
    conn.weights = conn.weights/conn.weights.max()
    conn.configure()
    return conn


conn_dopamine=np.load("conn_dopamine.npy")
conn_excitator=np.load("conn_excitator.npy")
conn_inhibitor=np.load("conn_inhibitor.npy")
conn = connectivity.Connectivity()
conn.region_labels = np.array(["L.BSTS", "L.CACG", "L.CMFG", "L.CU", "L.EC", "L.FG", "L.IPG", "L.ITG", "L.ICG","L.LOG", "L.LOFG", "L.LG", "L.MOFG", "L.MTG", "L.PHIG", "L.PaCG", "L.POP", "L.POR","L.PTR", "L.PCAL", "L.PoCG", "L.PCG", "L.PrCG", "L.PCU", "L.RACG", "L.RMFG", "L.SFG",
    "L.SPG", "L.STG", "L.SMG", "L.FP", "L.TP", "L.TTG", "L.IN", "L.CER", "L.TH", "L.CA","L.PU", "L.HI", "L.AC", "lh-GPe", "lh-GPi", "lh-STN", "rh-GPe","rh-GPi", "rh-STN", "R.TH", "R.CA", "R.PU", "R.HI",  "R.AC", "R.BSTS",
    "R.CACG", "R.CMFG", "R.CU", "R.EC", "R.FG", "R.IPG", "R.ITG", "R.ICG", "R.LOG","R.LOFG", "R.LG", "R.MOFG", "R.MTG", "R.PHIG", "R.PaCG", "R.POP", "R.POR", "R.PTR","R.PCAL", "R.PoCG", "R.PCG", "R.PrCG", "R.PCU", "R.RACG", "R.RMFG", "R.SFG", "R.SPG",
    "R.STG", "R.SMG", "R.FP", "R.TP", "R.TTG", "R.IN", "R.CER","SubstantiaNigraLH","SubstantiaNigraRH"])
conn.centres = np.zeros((3,88))
conn.weights = conn_excitator
conn.tract_lengths = (conn_excitator*.0+100).astype('i')
conn.configure()
# conn = [conn_dopamine, conn_excitator, conn_inhibitor]
mdl = models.MPRDopa()
g_dopa, g_excit, g_inhib = np.array([7e-1]), 7e-2, np.array([1.7e-2])

nsigma = 0.03
n_nodes = 88
r0, V0, u0, Sa0, Sg0, Dp0 = 0.1, -70.0, 0.0, 0.0, 0.0, 0.05
init_cond = np.array([r0, V0, u0, Sa0, Sg0, Dp0])
initial_cond = np.repeat(init_cond, n_nodes).reshape((1, 6, 88, 1))
sim = simulator.Simulator(
    model=mdl,
    connectivity=conn,
    coupling=coupling.Linear(
        a=np.array([g_excit])
    ),
    conduction_speed=1.,
            integrator=integrators.HeunStochastic(
            dt=0.01,
            noise=noise.Additive(
                nsig=np.array(
                    [nsigma,nsigma,nsigma,nsigma, nsigma,nsigma]
                ), 
                noise_seed=42)
        ),
    monitors=[monitors.Raw()],
    initial_conditions=initial_cond
).configure()

def run_dopa_backend(sim, *args, **kwargs):
    backend = DOPABackend()
    return backend.run_sim(sim, *args, **kwargs)

import time
start = time.time()
(t, buf), = run_dopa_backend(sim, conn_inhibitor, conn_dopamine, g_inhib, g_dopa, simulation_length=50000)
stop = time.time()
print(int(stop-start))

# sim explodes plot only the beginning
plt.plot(buf[0][:1000,0,:])
