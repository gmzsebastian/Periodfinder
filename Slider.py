import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons

#File = "J1407Out.txt"
#MinPeriod = 0.5
#MaxPeriod = 3.0

File = input('File with data (Time, Mag, Error):')
MinPeriod = input('Minimum Period (in hours):')
MaxPeriod = input('Maximum Period (in hours):')

Time0, Mag0, Error = np.genfromtxt(str(File), unpack = True)

Time = Time0
Mag = Mag0 / max(Mag0)
Error = Error / max(Mag0)

def phaseData(data, Period):
	Phase = np.zeros(len(data))
	for i in range(len(data)):
		Phase[i] = (24 * data[i] / Period) - np.int(24 * data[i] / Period)
	return Phase

fig, ax = plt.subplots()
plt.xlabel("Phase")
plt.ylabel("Magnitude")
plt.subplots_adjust(bottom=0.25)
BestPeriod = 5.52963 * 24.0
Phase = phaseData(Time, BestPeriod)
l, = plt.plot(Phase, Mag, lw=0, color='green', marker = 'o', markeredgewidth = 0)
plt.axis([0, 1, min(Mag)*0.99, max(Mag)*1.01])

axperiod = plt.axes([0.25, 0.1, 0.65, 0.03])

speriod = Slider(axperiod, 'Period', MinPeriod, MaxPeriod, valinit=BestPeriod)

# Change the value of the plot
def update(val):
    period = speriod.val
    l.set_xdata(phaseData(Time, period))
    fig.canvas.draw_idle()
speriod.on_changed(update)

# Reset parameters
resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
button = Button(resetax, 'Reset')

# Reset Button
def reset(event):
    speriod.reset()
button.on_clicked(reset)

plt.show()
