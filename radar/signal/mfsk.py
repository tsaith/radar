import numpy as np
from .waveforms import unit_square, staircase

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
    f_beat = v + R

    return f_beat

def get_phase_diff(R, v, t_step, f_offset):
    """
    Return the normalized pahse difference.

    R: normalized range.
    v: normalized velocity.
    t_step: normalized time duration of each step.
    f_offset: normalized frequency offset.
    """

    dphi = -t_step * v - f_offset * R

    return dphi

def range_resolution(f_sweep):

    light_speed = 3e8
    dR = light_speed / (2 * f_sweep)

    return dR

def phase_resolution(wave_length, t_chirp):

    dv = wave_length / (2 * t_chirp)

    return dv


def range_and_velocity(f_beat, dphi, t_step, f_offset):
    """
    Return the normalized range and velocity.
    f_beat: normalized beat frequency.
    dphi: normalized phase difference.
    t_step: normalized time duration of each step.
    f_offset: normalized frequency offset.
    """

    R = ( dphi + t_step*f_beat)   / (t_step - f_offset)
    v = (-dphi - f_offset*f_beat) / (t_step - f_offset)

    return R, v
