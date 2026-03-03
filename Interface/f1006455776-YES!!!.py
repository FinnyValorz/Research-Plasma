import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from dedalus import public as d3
from dedalus.extras.plot_tools import plot_bot
from scipy.special import expit as sigmoid

plt.clf()

### MOST UP TO DATE:

c = [1,1]
nu = 0
Lx = 2
interface = Lx/2
timestepper = d3.RK222


# Bases
xcoord = d3.Coordinate('x')
dist = d3.Distributor(xcoord, dtype=float)
x_Fbasis = d3.Chebyshev(xcoord, 64, bounds=(0,interface), dealias=1)
x_Sbasis = d3.Chebyshev(xcoord, 64, bounds=(interface, Lx), dealias=1)

# Substitutions
dx = lambda A: d3.Differentiate(A, xcoord)
int = lambda A: d3.Integrate(A, xcoord).evaluate()['g'][0]

x_F = dist.local_grids(x_Fbasis, scales=1)[0]
lift_basis_F = x_Fbasis.derivative_basis(1)
lift_F = lambda A: d3.Lift(A, lift_basis_F, -1)

x_S = dist.local_grids(x_Sbasis, scales=1)[0]
lift_basis_S = x_Sbasis.derivative_basis(1)
lift_S = lambda A: d3.Lift(A, lift_basis_S, -1)

compound = np.append(x_F, x_S)

# Fields
phi_F = dist.Field(bases=x_Fbasis, name='phi_F')
pi_F  = dist.Field(bases=x_Fbasis, name='pi_F')
zeta_F = dist.Field(bases=x_Fbasis, name='zeta_F')
tau1_F = dist.Field(name='tau1_F')
tau2_F = dist.Field(name='tau2_F')

phi_S = dist.Field(bases=x_Sbasis, name='phi_S')
pi_S  = dist.Field(bases=x_Sbasis, name='pi_S')
zeta_S = dist.Field(bases=x_Sbasis, name='zeta_S')
tau1_S = dist.Field(name='tau1_S')
tau2_S = dist.Field(name='tau2_S')

c_F = dist.Field(bases=x_Fbasis, name='c_F')
c_S = dist.Field(bases=x_Sbasis, name='c_S')
c_F['g'] = (c[1]-c[0])*sigmoid(80*(x_F-interface)) + c[0]
c_S['g'] = (c[1]-c[0])*sigmoid(80*(x_S-interface)) + c[0]

H_F = 0.5*pi_F**2 + 0.5*c_F**2*dx(phi_F)**2
H_S = 0.5*pi_S**2 + 0.5*c_S**2*dx(phi_S)**2
dHdt = pi_S(x=2)*c_S(x=2)**2*dx(phi_S)(x=2) - pi_F(x=0)*c_F(x=0)**2*dx(phi_F)(x=0)
dHdt_F = dx(pi_F*c_F**2*dx(phi_F))
dHdt_S = dx(pi_S*c_S**2*dx(phi_S))
Htau_F = pi_F*(c_F**2*dx(lift_F(tau2_F)) + lift_F(tau1_F))
Htau_S = pi_S*(c_S**2*dx(lift_S(tau2_S)) + lift_S(tau1_S))

#boundary = dist.Field(name='boundary')

# Problem
vars = [phi_F, pi_F, zeta_F, tau1_F, tau2_F, phi_S, pi_S, zeta_S, tau1_S, tau2_S]
problem = d3.IVP(vars, namespace=locals())

problem.add_equation("dt(pi_F) + 2*nu*pi_F - dx(c_F**2*zeta_F) - lift_F(tau1_F) = 0")
problem.add_equation("dt(phi_F) - pi_F = 0")
problem.add_equation("zeta_F - dx(phi_F) - lift_F(tau2_F) = 0")

problem.add_equation("dt(pi_S) + 2*nu*pi_S - dx(c_S**2*zeta_S) - lift_S(tau1_S) = 0")
problem.add_equation("dt(phi_S) - pi_S = 0")
problem.add_equation("zeta_S - dx(phi_S) - lift_S(tau2_S) = 0")

problem.add_equation("phi_F(x=0) = 0")
problem.add_equation("phi_F(x=1) - phi_S(x=1) = 0")
problem.add_equation("c_F(x=1)**2*dx(phi_F)(x=1) - c_S(x=1)**2*dx(phi_S)(x=1) = 0")
problem.add_equation("phi_S(x=Lx) = 0")

# Solver
solver = problem.build_solver(timestepper)
solver.stop_sim_time = 15

# Initial Conditions
phi_F['g'] = np.sin(np.pi/Lx * x_F) / (Lx-x_F + 0.1)
phi_S['g'] = np.sin(np.pi/Lx * x_S) / (Lx-x_S + 0.1)


graphF = np.append(phi_F['g'], [None for i in range(len(x_S))])
graphS = np.append([None for i in range(len(x_F))], phi_S['g'])
#plt.plot(compound, graphF)
#plt.plot(compound, graphS)
plt.plot(compound, [(c[1]-c[0])*sigmoid(80*(x-interface))+c[0] for x in compound]) 
#plt.show()
plt.close()

# Main Loop
phi_result = [np.append(phi_F['g'], phi_S['g'])]
t_list = [float(solver.sim_time)]
E_list = [int(H_F) + int(H_S)]
Etau_list = [int(Htau_F) + int(Htau_S)]
EF_list = [int(H_F)]
ES_list = [int(H_S)]
dEdt_list = [dHdt['g'][0]]
dEdt_list = [int(dHdt_F) + int(dHdt_S)]

timestep = 0.01
# Norman says to pick CFL appropriate step size = Lx/# of grid points.
while solver.proceed:
    solver.step(timestep)
    phi_result.append(np.append(np.copy(phi_F['g']), np.copy(phi_S['g'])))
    t_list.append(float(solver.sim_time))
    E_list.append(int(H_F) + int(H_S))
    Etau_list.append(int(Htau_F) + int(Htau_S))
    dEdt_list.append(dHdt['g'][0])
    EF_list.append(int(H_F))
    ES_list.append(int(H_S))
    #dEdt_list.append(int(dHdt_F) + int(dHdt_S))
phi_result = np.array(phi_result)


# Plotting
# Phi Animation
fig, ax = plt.subplots()
line, = ax.plot(compound, phi_result[0])
ax.set_ylim(-2, 2)
ax.set_xlabel("$x$")
ax.set_ylabel("$\phi$")
def animate(t):
    line.set_ydata(phi_result[t])
    return line,

ani = animation.FuncAnimation(fig, animate, frames=len(phi_result), interval=100, blit=True)
# ani.save("movie.mp4")
plt.show()
plt.close()

# Energy Plot
def timederiv(list, n=1):
    if n==0:
        return list
    result = [(list[i+1] - list[i]) / (t_list[i+1] - t_list[i]) for i in range(len(list)-1)]
    return timederiv(result, n-1)
dEdt_list = timederiv(E_list)
d2Edt2_list = timederiv(E_list, 2)
E_T = [E_list[0] for t in t_list]
#plt.plot(t_list, E_T)
plt.plot(t_list, E_list)
#plt.plot(t_list[1::], dEdt_list)
#plt.plot(t_list[2::], d2Edt2_list)
#plt.plot(t_list, Etau_list)
plt.show()
plt.close()


# Phi Surface Plot
X, T = np.meshgrid(compound, t_list)
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot_surface(X, T, phi_result)
ax.set_ylim(0,5)
ax.set_xlabel('$x$')
ax.set_ylabel('$t$')
ax.set_zlabel('$\phi$')
#plt.show()
plt.close()






