import numpy as np
from radar.signal.mfsk import Rv_from_signals, fake_sampled_signals

# Simulation parameters
t_chirp = 8e-3 # Chirp time
num_steps = 512 # Number of steps, it should be even
num_pairs = round(num_steps / 2) # Number of pairs
t_step = t_chirp / num_steps # Time duration for each step
f_sweep = 150e6 # Sweep freq range
f_incr = f_sweep / (0.5*num_steps-1)
f_offset = -0.5*f_incr
f_carrier = 24e9 # Carrier frequency

# Target range and velocity
R_arr = np.array([30, 60, 40])
v_arr = np.array([60, 30, -20])

# Fake sampled signals
signal_a, signal_b = fake_sampled_signals(R_arr, v_arr, 
    num_steps, t_step, f_sweep, f_carrier)

# Predicted range and velocity
R_pred_arr, v_pred_arr = Rv_from_signals(signal_a, signal_b, 
    num_steps, t_chirp, t_step, f_sweep, f_offset, f_carrier)

print("Actual R = ", R_arr)
print("Actual v = ", v_arr)
print("Predicted R = ", R_pred_arr)
print("Predicted v = ", v_pred_arr)
