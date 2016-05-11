import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op
import emcee
import corner
import datetime
from scipy import optimize
Start = datetime.datetime.now()

# Import the Spectral Data
x, y, yerr = np.genfromtxt("J1407Flat.txt", unpack = True)

# Multiply the flux by a constant to get reasonable numbers
y = y * 1E18
yerr = yerr * 1E18

# Choose the initial gues.
h_guess  = 20.0         # Lines Height
c1_guess = 6563.0 - 8   # Line1 Center
c2_guess = 6563.0 + 8   # Line2 Center
w_guess  = 7.0          # Lines Width
f_true   = -1.0         # Overestimation of Error Bars

# Define a Gaussian function
def gaussian(x, h, c, w):
    return h*np.exp(-(x - c)**2/(2*w**2))

# Define a double double Gaussian
def doublegauss(x, h1, c1, c2, w1):
    return (gaussian(x, h1, c1, w1) + gaussian(x, h1, c2, w1))

# Error Function for the Double Gaussian
errfunc2 = lambda p, x, y: (doublegauss(x, *p) - y)**2
FirstGuess = [h_guess, c1_guess, c2_guess, w_guess]

# Do a quick least squares fit to the data
Optimized, success = optimize.leastsq(errfunc2, FirstGuess[:], args=(x, y))
# Save the output to use in MCMC
h_true  = Optimized[0]
c1_true = Optimized[1]
c2_true = Optimized[2]
w_true  = Optimized[3]

# Log of probabilty function
def logprob(theta, x, y, yerr):
    h, c1, c2, w, lnf = theta
    model = gaussian(x, h, c1, w) + gaussian(x, h, c2, w)
    sn2 = yerr**2 + model**2*np.exp(2*lnf)
    toSum = (y - model) ** 2 / sn2 + np.log( 2 * np.pi * sn2)
    return - 0.5 * np.sum(toSum)

# Informed guess after least square fits
Guess1 = np.array([h_true, c1_true, c2_true, w_true, f_true])

# Log prior funtion, define the priors here
def lnprior(theta):
    h, c1, c2, w, lnf = theta
    if 5 < h < 30 and 6550.0 < c1 < 6562.0 and 6562.0 < c2 < 6575.0 and 5.0 < w < 20.0 and -5 < lnf < 1:
        return 0.0
    return -np.inf

# Probability function
def lnprob(theta, x, y, yerr):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + logprob(theta, x, y, yerr)

# Number of parameters being fit, number of walkers, stepsize
ndim, nwalkers, stepsize = 5, 500, 1E-3
pos = [Guess1 + stepsize*np.random.randn(ndim) for i in range(nwalkers)]

# Run the MCMC
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(x, y, yerr))
print "Runing MCMC"
sampler.run_mcmc(pos, 2000)
samples = sampler.chain[:, 50:, :].reshape((-1, ndim))

# Plot the triangle plot
print "Plotting"
fig = corner.corner(samples, labels=["$h$", "$c1$","$c2$", "$w$", "$f$"], truths=[h_true, c1_true, c2_true, w_true, f_true])
fig.savefig("Triangle_All.png")
plt.clf()

# Obtain the parametrs of the best fit
h_mcmc, c1_mcmc, c2_mcmc, w_mcmc, f_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                                           zip(*np.percentile(samples, [16, 50, 84],
                                           axis=0)))

##### Plotting ######

Norm = 2 * np.sqrt(2 * np.log(2))
# Best fit parametrs, strings
A = str(np.around(h_mcmc[0], decimals=3))
B1 = str(np.around(c1_mcmc[0], decimals=3))
B2 = str(np.around(c2_mcmc[0], decimals=3))
C = str(np.around(Norm * w_mcmc[0], decimals=3))
D = str(np.around(f_mcmc[0], decimals=3))
E = str(np.around(c2_mcmc[0] - c1_mcmc[0], decimals=3))

# Best fit parametrs errors, strings
Aerr = str(np.around(h_mcmc[1], decimals=3))
B1err = str(np.around(c1_mcmc[1], decimals=3))
B2err = str(np.around(c2_mcmc[1], decimals=3))
Cerr = str(np.around(Norm * w_mcmc[1], decimals=3))
Derr = str(np.around(f_mcmc[1], decimals=3))
Eerr = str(np.around(np.sqrt(c1_mcmc[1] **2 + c2_mcmc[1] **2), decimals=3))

# Parameters common for both plots
def ToPlot(Name):
    # Overal figure parameters
    plt.subplots_adjust(left=0.1, right=0.95, top=0.92, bottom=0.1)
    plt.xticks(size = 18); plt.yticks(size = 18)
    plt.xlim(min(x), max(x))

    # Names and titles
    plt.xlabel("Wavelength [A]", fontsize = 18)
    plt.ylabel("Scaled Flux", fontsize = 18)
    plt.title("J1407   " + 
        "H = " + A + "+/-" + Aerr +
        "   D = " + E + "+/-" + Eerr + "\n" + 
        "   W = " + C + "+/-" + Cerr +
        "   f = " + D + "+/-" + Derr, fontsize = 18 )
    plt.legend(loc = "best", fontsize = 18)
    plt.savefig(Name, dpi = 500)
    plt.clf()

# Plot the figure with all the different fits
plt.plot(x, y, lw = 1, c = 'g', label = 'Data')
plt.plot(x, doublegauss(x, h_guess, c1_guess, c2_guess, w_guess), c = 'k', label = 'Guess')
plt.plot(x, doublegauss(x, *Optimized), lw=1, c='r', ls='--', label='Optimized')
plt.plot(x, doublegauss(x, h_mcmc[0], c1_mcmc[0], c2_mcmc[0], w_mcmc[0]), c = 'b', label = 'MCMC Fit')
ToPlot("Fit_All.jpg")

# Plot the final figure with only data and best fit
plt.plot(x, y, lw = 1, c = 'g', label = 'Data')
plt.errorbar(x, y, yerr= yerr / f_mcmc[0], fmt = 'o', c = 'g')
plt.plot(x, doublegauss(x, h_mcmc[0], c1_mcmc[0], c2_mcmc[0], w_mcmc[0]), c = 'b', label = 'MCMC Fit')
ToPlot("Fit_All+Err.jpg")

End = datetime.datetime.now()

print "It Took = " + str(End - Start)


