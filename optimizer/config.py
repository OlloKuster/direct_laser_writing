# Not a class since I don't need to pass it differently according to the problem.
import nlopt

MAXEVAL = 10
FTOL_ABS = 1e-7
FTOL_REL = 1e-5
UPPER_BOUNDS = 1
LOWER_BOUNDS = 0
OPTIMISER = nlopt.LD_MMA   # MMA for lens, LBFGS for cross

cur_it = 0
ind = 0