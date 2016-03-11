import numpy as np
from scipy import signal

def unit_square(t, T):
    """
    Return unit square waves.

    t: Time array.
    T: Time Period.
    """

    y = 0.5*(signal.square(2.0 * np.pi * t / T) + 1.0)

    return y

def staircase(t, T, height):
    """
    Return waves of staircase.

    t: Time array.
    T: Time Period.
    height: stair height.
    """

    eps = 1e-12 # Error tolerance

    y = height * unit_square(t, T)
    m = ((t+eps) / T).astype(int)
    y = m*height

    return y

def fm_waves(f, dt, amp0=1, phi0=0):
  """
  Return the sinusoidal waves with frequency modulation.

  y(t) = amp0 * cos(phi(t) + phi0),
  where phi(t) = 2 * pi * \int_0^t f(t) * dt.

  f: Instantaneous frequencies.
  dt: Time interval.
  amp0: Amplitude.
  phi0: Initail phase. When it is -pi/2, sin waves are produced.
  """

  phi = 2 * np.pi * np.cumsum(f) * dt
  y = amp0*np.cos(phi+ phi0)

  return y

def sinusoidal(f, t, amp0=1, phi0=0):
  """
  Return the sinusoidal wave with constant frequency.

  f: Instantaneous frequencies.
  dt: Time interval.
  amp0: Amplitude.
  phi0: Initail phase. When it is -pi/2, sin waves are produced.
  """

  phi = 2 * np.pi * f * t
  y = amp0*np.cos(phi + phi0)

  return y

def square_freq(t, fm=1, B=1, fd=0, duty=0.5):
  '''
  Simulated frequencies with square modulation.

  t: Time array.
  fm: Modulation frequency.
  fd: Doppler frequency shift.
  B: Bandwidth.
  '''
  f = B*0.5*(signal.square(2 * np.pi * fm * t, duty=duty) + 1)
  f += fd

  return f

def sawtooth_freq(t, fm=1, B=1, fd=0, width=0.5):
  '''
  Simulated frequencies of sawtooth modulation.

  t: Time array.
  fm: Modulation frequency.
  fd: Doppler frequency shift.
  B: Bandwidth.
  '''
  f = B*0.5*(signal.sawtooth(2 * np.pi * fm * t, width=width) + 1)
  f += fd

  return f
