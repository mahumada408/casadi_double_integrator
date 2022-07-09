# MPC for a multivariable system.
import numpy as np
import matplotlib.pyplot as plt
import mpctools as mpc
import mpctools.plots as mpcplots

# Define continuous time model.
Ts = 0.01
Acont = np.array([[1., Ts], [0., 1.]]) # states are [position; velocity]
Bcont = np.array([[0.5*Ts**2], [Ts]])  # control [acceleration]
n = Acont.shape[0] # Number of states.
m = Bcont.shape[1] # Number of control elements

# Discretize.
dt = .025
Nt = 20
(A, B) = mpc.util.c2d(Acont,Bcont,Ts)
def ffunc(x,u):
    """Linear discrete-time model."""
    return mpc.mtimes(A, x) + mpc.mtimes(B, u)
f = mpc.getCasadiFunc(ffunc, [n, m], ["x", "u"], "f")

# Bounds on u.
umax = 2
lb = dict(u=[-umax])
ub = dict(u=[umax])

# Define Q and R matrices.
Q = 100000*np.eye(n)
R = 0.1*np.eye(m)
def l(x, u, x_sp, u_sp):
    """Stage cost with setpoints."""
    dx = x - x_sp
    du = u
    return mpc.mtimes(dx.T, Q, dx) + mpc.mtimes(du.T, R, du)
    # return mpc.mtimes(dx.T, Q, dx)
lcasadi = mpc.getCasadiFunc(l, [n, m, n, m], ["x","u","x_sp","u_sp"], "l")

# Initial condition and sizes.
x0 = np.array([0,0])
N = {"x" : n, "u" : m, "t" : Nt}
funcargs = {"f" : ["x","u"], "l" : ["x","u","x_sp","u_sp"]}

# Now simulate.
sp = { "x": np.array([0.2,0]), "u": np.array([0])}
solver = mpc.nmpc(f, lcasadi, N, x0, lb, ub, sp=sp, funcargs=funcargs, verbosity=0, isQP=True)
des_time = 50 # s
nsim = int(des_time / Ts)
t = np.arange(nsim+1)*dt
xcl = np.zeros((n,nsim+1))
xcl[:,0] = x0
ucl = np.zeros((m,nsim))
for k in range(nsim):
    solver.fixvar("x", 0, x0)
    sol = mpc.callSolver(solver)
    print("Iteration %d Status: %s" % (k, sol["status"]))
    print(x0)
    print(sol["u"][0])
    print(ucl[:,k])
    xcl[:,k] = x0
    ucl[:,k] = sol["u"][0,:]
    x0 = ffunc(x0, sol["u"][0]) # Update x0.
xcl[:,nsim] = x0 # Store final state.

plt.subplot(3,1,1)
plt.plot(xcl[:,0])
plt.subplot(3,1,2)
plt.plot(xcl[:,1])
plt.subplot(3,1,3)
plt.plot(ucl[:,0])
plt.savefig('double_integrator_closed_loop.pdf')  

# # Plot things. Since time is along the second dimension, we must specify
# # timefirst = False.
# fig = mpc.plots.mpcplot(xcl,ucl,t,np.zeros(xcl.shape),xinds=[0],
#                         timefirst=False)
# mpcplots.showandsave(fig, "double_integrator_closed_loop.pdf")
