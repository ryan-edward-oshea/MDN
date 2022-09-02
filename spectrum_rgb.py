"""
Based on 
  http://www.physics.sfasu.edu/astro/color/spectra.html
  RGB VALUES FOR VISIBLE WAVELENGTHS   by Dan Bruton (astro@tamu.edu)
"""

import numpy as np
select=np.select
power=np.power
transpose=np.transpose
arange=np.arange

def factor(wl):
    return select(
        [ wl > 700.,
          wl < 420.,
          True ],
        [ .3+.7*(780.-wl)/(780.-700.),
          .3+.7*(wl-380.)/(420.-380.),
          1.0 ] )

def raw_r(wl):
    return select(
        [ wl >= 580.,
          wl >= 510.,
          wl >= 440.,
          wl >= 380.,
          True ],
        [ 1.0,
          (wl-510.)/(580.-510.),
          0.0,
          (wl-440.)/(380.-440.),
          0.0 ] )

def raw_g(wl):
    return select(
        [ wl >= 645.,
          wl >= 580.,
          wl >= 490.,
          wl >= 440.,
          True ],
        [ 0.0,
          (wl-645.)/(580.-645.),
          1.0,
          (wl-440.)/(490.-440.),
          0.0 ] )

def raw_b(wl):
    return select(
        [ wl >= 510.,
          wl >= 490.,
          wl >= 380.,
          True ],
        [ 0.0,
          (wl-510.)/(490.-510.),
          1.0,
          0.0 ] )

gamma = 0.80
def correct_r(wl):
    return power(factor(wl)*raw_r(wl),gamma)
def correct_g(wl):
    return power(factor(wl)*raw_g(wl),gamma)
def correct_b(wl):
    return power(factor(wl)*raw_b(wl),gamma)

ww=arange(380.,781.)
def get_spectrum_cmap():
    ''' use like: 
      cmap = get_spectrum_cmap()
      plot(..., color=[cmap.to_rgba(nm) for nm in wavelengths])
    '''
    from matplotlib import colors, cm
    cmap = transpose([correct_r(ww),correct_g(ww),correct_b(ww)])
    return cm.ScalarMappable(norm=colors.Normalize(vmin=380, vmax=780),
                         cmap=colors.ListedColormap(cmap))



''' 
Color format conversion helpers
https://gist.github.com/mathebox/e0805f72e7db3269ec22
- not correct: hsl_to_rgb( rgb_to_hsl(x) )
- may be others
'''
def rgb_to_hsv(r, g, b):
    r = float(r)
    g = float(g)
    b = float(b)
    high = max(r, g, b)
    low = min(r, g, b)
    h, s, v = high, high, high

    d = high - low
    s = 0 if high == 0 else d/high

    if high == low:
        h = 0.0
    else:
        h = {
            r: (g - b) / d + (6 if g < b else 0),
            g: (b - r) / d + 2,
            b: (r - g) / d + 4,
        }[high]
        h /= 6

    return h, s, v

def hsv_to_rgb(h, s, v):
    i = math.floor(h*6)
    f = h*6 - i
    p = v * (1-s)
    q = v * (1-f*s)
    t = v * (1-(1-f)*s)

    r, g, b = [
        (v, t, p),
        (q, v, p),
        (p, v, t),
        (p, q, v),
        (t, p, v),
        (v, p, q),
    ][int(i%6)]

    return r, g, b

def rgb_to_hsl(r, g, b):
    r = float(r)
    g = float(g)
    b = float(b)
    high = max(r, g, b)
    low = min(r, g, b)
    h, s, l = ((high + low) / 2,)*3

    if high == low:
        h = 0.0
        s = 0.0
    else:
        d = high - low
        s = d / (2 - high - low) if l > 0.5 else d / (high + low)
        h = {
            r: (g - b) / d + (6 if g < b else 0),
            g: (b - r) / d + 2,
            b: (r - g) / d + 4,
        }[high]
        h /= 6

    return h, s, l

def hsl_to_rgb(h, s, l):
    def hue_to_rgb(p, q, t):
        t += 1 if t < 0 else 0
        t -= 1 if t > 1 else 0
        if t < 1/6: return p + (q - p) * 6 * t
        if t < 1/2: return q
        if t < 2/3: p + (q - p) * (2/3 - t) * 6
        return p

    if s == 0:
        r, g, b = l, l, l
    else:
        q = l * (1 + s) if l < 0.5 else l + s - l * s
        p = 2 * l - q
        r = hue_to_rgb(p, q, h + 1/3)
        g = hue_to_rgb(p, q, h)
        b = hue_to_rgb(p, q, h - 1/3)

    return r, g, b

def hsv_to_hsl(h, s, v):
    l = 0.5 * v  * (2 - s)
    s = v * s / (1 - math.fabs(2*l-1))
    return h, s, l

def hsl_to_hsv(h, s, l):
    v = (2*l + s*(1-math.fabs(2*l-1)))/2
    s = 2*(v-l)/v
    return h, s, v