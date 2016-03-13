import numpy as np
from .waveforms import unit_square, staircase, sinusoidal

def transmitted_freq(t, num_steps, t_step, f_sweep, f_carrier, f_offset=None):
    """
    Return the frequencies of the transmitted signal.
    """

    f_incr = f_sweep / (0.5*num_steps-1)

    if f_offset == None:
        f_offset = -0.5 * f_incr

    f1 = staircase(t, 2*t_step, height=f_incr)
    f2 = f_offset * (1 - unit_square(t, 2*t_step))

    f = f1 + f2
    f += f_carrier

    return f

def received_freq(f_t, t_delay, dt, fd):
    """
    Return the frequencies of the received signal.
    """

    num_time_shift = int(t_delay / dt)

    f_r = f_t.copy()
    f_r += fd

    f_r = np.roll(f_r, num_time_shift)
    f_r[:num_time_shift] = 0

    return f_r

def doppler_shift(v, f):
    """
    Return the frequency shift due to the Doppler effect.

    v: relative velocity
    f: frequency.
    """
    light_speed = 3e8
    fd = 2.0*v/light_speed*f
    return fd

def get_beat_freq(R, v):
    """
    Return the normalized beat frequency.

    R: normalized range.
    v: normalized velocity.
    """
    f_beat = v - R

    return f_beat

def get_phase_diff(R, v, t_step, f_offset):
    """
    Return the normalized pahse difference.

    R: normalized range.
    v: normalized velocity.
    t_step: normalized time duration of each step.
    f_offset: normalized frequency offset.
    """

    dphi = t_step * v - f_offset * R

    return dphi


def range_and_velocity(f_beat, dphi, t_step, f_offset):
    """
    Return the normalized range and velocity.
    f_beat: normalized beat frequency.
    dphi: normalized phase difference.
    t_step: normalized time duration of each step.
    f_offset: normalized frequency offset.
    """

    R = (dphi - t_step*f_beat)   / (t_step - f_offset)
    v = (dphi - f_offset*f_beat) / (t_step - f_offset)

    return R, v

def range_resolution(f_sweep):

    light_speed = 3e8
    dR = light_speed / (2 * f_sweep)

    return dR

def phase_resolution(wave_length, t_chirp):

    dv = wave_length / (2 * t_chirp)

    return dv

def freq_phase_analytic(R, v, num_steps, t_step, f_sweep, f_carrier):
    """
    Returm the normalized beat frequency and phase change.
    """

    # Physical constants
    light_speed = 3e8 # Light speed (m/s)
    pi2 = 2*np.pi

    # Simulation parameters
    num_pairs = num_steps / 2 # Number of pairs
    t_chirp = num_steps * t_step
    f_incr = f_sweep / (0.5*num_steps-1)
    f_offset = -0.5*f_incr
    wave_length = light_speed / f_carrier
    t_delay = 2*R/light_speed
    fd = doppler_shift(v, f_carrier)
    dR = range_resolution(f_sweep)              # Range resolution
    dv = phase_resolution(wave_length, t_chirp) # Velocity resolution

    # Normalized varaibles
    R_n = R / dR
    v_n = v / dv
    t_step_n = t_step / t_chirp
    f_offset_n = f_offset / f_sweep

    # normalized beat freq.
    f_beat_n = get_beat_freq(R_n, v_n) # normalized beat freq.# normalized phase difference
    dphi_n = get_phase_diff(R_n, v_n, t_step_n, f_offset_n)
    f_beat = f_beat_n / t_chirp
    dphi = dphi_n * pi2

    return f_beat_n, dphi_n

def multi_freq_phase_analytic(R_arr, v_arr, num_steps, t_step, f_sweep, f_carrier):
    """
    Return multiple normalized beat freq and phase change.

    R: Range array.
    v: Velocity array.
    """

    assert len(R_arr) == len(v_arr)
    num_targets = len(R_arr)

    f_beat_arr = []
    dphi_arr = []
    for i in range(num_targets):
        R = R_arr[i]
        v = v_arr[i]
        f_beat, dphi = freq_phase_analytic(R, v, num_steps, t_step, f_sweep, f_carrier)

        f_beat_arr.append(f_beat)
        dphi_arr.append(dphi)

    f_beat_arr = np.asarray(f_beat_arr)
    dphi_arr = np.asarray(dphi_arr)

    return f_beat_arr, dphi_arr

def fake_sampled_signals(R_arr, v_arr, num_steps, t_step, f_sweep, f_carrier):
    """
    Return the sampled signals of A and B sequences.
    """
    num_targets = R_arr.size
    num_pairs = round(num_steps / 2)

    t_chirp = num_steps * t_step

    # Beat freq. and phase change
    f_beat_n_arr, dphi_n_arr = multi_freq_phase_analytic(R_arr, v_arr, num_steps, t_step, f_sweep, f_carrier)
    f_beat_arr = f_beat_n_arr / t_chirp
    dphi_arr = dphi_n_arr * 2 * np.pi

    # Time array
    num_samp = num_pairs
    t = np.linspace(0, t_chirp, num_samp, endpoint=False)

    # Amplitude
    amp_arr = 1 / (2*R_arr)**2
    amp_arr /= np.max(amp_arr)

    # Create the fake signals of sweep A and B
    signal_a = 0
    signal_b = 0
    for i in range(num_targets):
        signal_a += sinusoidal(f_beat_arr[i], t, amp0=amp_arr[i], phi0=0)
        signal_b += sinusoidal(f_beat_arr[i], t, amp0=amp_arr[i], phi0=dphi_arr[i])

    return signal_a, signal_b
