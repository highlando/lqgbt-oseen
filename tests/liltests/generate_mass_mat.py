import dolfin
import dolfin_navier_scipy.dolfin_to_sparrays as dts

N = 20

mesh = dolfin.UnitSquareMesh(N, N)
V = dolfin.FunctionSpace(mesh, 'CG', 1)
u = dolfin.TrialFunction(V)
v = dolfin.TestFunction(V)

mass = dolfin.assemble(v*u*dolfin.dx)
M = dts.mat_dolfin2sparse(mass)
