# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Gianmarco Pinton
"""FSA beamforming via MACH (mach-beamform) + sector scan conversion.

Uses the open-source MACH GPU beamformer (github.com/Forest-Neurotech/mach,
Clear BSD) -- an OPTIONAL extra (``pip install fullwave2-ultra[beamform]``).
``scan_convert`` and ``_bandpass`` are pure numpy/scipy and import without MACH
or a GPU; only :func:`beamform_fsa` (and its ``_mach_compound`` helper) need the
``mach`` package + cupy at call time.

Implements the FSA pipeline: per-transmit genout -> average sub-elements per
element -> data-derived t0 (self-receive envelope peak) -> mach coherent
compound over the single-element transmits -> envelope -> log-compress ->
scan_convert (polar sector -> Cartesian).
"""
from __future__ import annotations
import numpy as np
from . import io_dat
# scipy.signal / scipy.interpolate are imported lazily inside the functions that
# use them, so `import fullwave2_ultra.beamform` (e.g. for scan_convert only) stays
# light and does not pull scipy.signal at module load.


# ---- scan conversion -----------------------------------------------------
def scan_convert(img, r, theta, beam_orig, apex, spacing):
    """img (R,A) in (r,theta) -> Cartesian sector. Returns (sector, x, z)."""
    from scipy.interpolate import RegularGridInterpolator
    r = np.asarray(r).ravel(); theta = np.asarray(theta).ravel()
    beam_orig = np.atleast_2d(beam_orig); apex = np.asarray(apex).ravel()
    th_m, r_m = np.meshgrid(theta, r)                       # (R,A)
    ox = beam_orig[:, 0][None, :]; oz = beam_orig[:, -1][None, :]
    X = ox + r_m * np.sin(th_m); Z = oz + r_m * np.cos(th_m)
    if np.isscalar(spacing): spacing = (spacing, spacing)
    Xq = np.arange(X.min(), X.max(), spacing[0]); Zq = np.arange(Z.min(), Z.max(), spacing[1])
    Xqm, Zqm = np.meshgrid(Xq, Zq)
    rad_off = np.mean(np.linalg.norm(beam_orig[:, [0, -1]] - apex[[0, -1]], axis=1))
    Rq = np.sqrt((Xqm - apex[0]) ** 2 + (Zqm - apex[-1]) ** 2) - rad_off
    Aq = np.arctan2(Xqm - apex[0], Zqm - apex[-1])
    f = RegularGridInterpolator((r, theta), img, method="cubic",
                                bounds_error=False, fill_value=np.nan)
    sector = f(np.stack([Rq.ravel(), Aq.ravel()], -1)).reshape(Xqm.shape)
    return sector, Xq - np.mean(beam_orig[:, 0]), Zq - np.max(beam_orig[:, -1])


def _bandpass(rf, fs, f0, bw):
    from scipy.signal import firwin, filtfilt
    lo, hi = f0 * (1 - bw / 2), f0 * (1 + bw / 2)
    ny = fs / 2
    taps = firwin(65, [lo / ny, min(hi / ny, 0.99)], pass_zero=False)
    return filtfilt(taps, 1.0, rf, axis=0)


def _mach_compound(cd, rx, scan, arrivals, *, rx_start_s, fs, fnum, c0, modfreq, interp,
                   tukey_alpha=0.5):
    """Coherent-compound DAS over transmits via MACH, forwarding interp_type
    (mach.experimental.beamform does not expose it). cd: (n_tx, n_rx, n_samp, n_frames),
    arrivals: (n_tx, n_scan). Accumulates each transmit into one output (MACH's
    kernel.beamform sums into `out`)."""
    from mach import kernel
    from mach.kernel import InterpolationType
    itype = {"nearest": InterpolationType.NearestNeighbor,
             "linear": InterpolationType.Linear,
             "quadratic": InterpolationType.Quadratic}[interp]
    try:
        import cupy as xp
    except ImportError:
        xp = np
    n_tx, n_scan, n_fr = cd.shape[0], scan.shape[0], cd.shape[3]
    out = xp.zeros((n_scan, n_fr), dtype=xp.complex64)
    rxg, scang = xp.asarray(rx), xp.asarray(scan)
    for n in range(n_tx):
        kernel.beamform(xp.asarray(cd[n]), rxg, scang, xp.asarray(arrivals[n]),
                        out=out, rx_start_s=rx_start_s, sampling_freq_hz=fs, f_number=fnum,
                        sound_speed_m_s=c0, modulation_freq_hz=modfreq, interp_type=itype,
                        tukey_alpha=tukey_alpha)
    return out.get() if hasattr(out, "get") else np.asarray(out)


def beamform_fsa(rundir, meta, *, fnum=2.0, bw=0.7, backend="mach", dbrange=60,
                 mod_freq_sign=1, txfnum=None, channel_data=None,
                 convention="iq", interp="linear", tukey_alpha=0.5, normalize="sum"):
    """FSA-beamform a run dir's genout into a sector B-mode (dict with b_sector,
    x, z, env_polar, bmode_db, angles, rs).

    fnum     receive dynamic-aperture f-number (MACH applies it on receive).
    txfnum   optional TRANSMIT acceptance-cone f-number (RTBF-style). Default None
             compounds ALL transmits per pixel (brighter, flatter-with-depth, denser
             speckle). Set e.g. txfnum=2.0 to gate each pixel to transmits within a
             half-aperture depth/(2*txfnum) -- a dimmer, steeper, grainier image that
             matches the legacy RTBF (txfnum=rxfnum=2). MACH has no mask argument, so
             the cone is imposed by pushing out-of-cone (transmit,pixel) arrivals out
             of the sample range (the kernel marks those samples invalid -> zero
             contribution); equivalent to a boxcar transmit apodization.
    channel_data  optional (nsims, nframes, npx) real per-element RF, used as-is
             (skips genout read + bandpass) and demodulated to baseband IQ for MACH.
             Handy for feeding externally-prepared channels (e.g. a reference
             pipeline's) to isolate beamformer vs data-path differences.
    convention  'iq' (default): demodulate to baseband IQ and let MACH re-impose the
             carrier (modulation_freq_hz=+f0) -- robust to the coarse fs/f0~4.4
             sample rate. 'analytic_rf': feed the analytic signal (Hilbert, carrier
             kept) with modulation_freq_hz=0 -- pure delay-and-sum of the oscillating
             analytic samples, matching the legacy RTBF (Hilbert-then-focus) path.
    interp   'linear' (default) | 'quadratic' | 'nearest' -- MACH sample interpolation.
             analytic_rf benefits from 'quadratic' (interpolating an oscillating signal).
    tukey_alpha  receive-aperture apodization window (MACH default 0.5). Set 0.0 for a
             boxcar/rectangular aperture -- matches RTBF's hard f# cutoff (grainier speckle).
    normalize  'sum' (default, MACH native: brighter, flatter-with-depth) or 'mean'
             (divide each pixel by its contributing tx*rx count -- matches RTBF's
             sum/sumcount, which dims the image and steepens the depth rolloff).
             RTBF-like look: convention='iq', tukey_alpha=0.0, normalize='mean', txfnum=fnum.
    """
    if backend != "mach":
        raise NotImplementedError("only the MACH backend is implemented")
    from scipy.signal import hilbert

    xdc = meta["xdc"]
    dX = meta["dX"]; dY = meta.get("dY", dX); dT = meta["dT"]
    fs = meta["fs"]; f0 = meta["f0"]; c0 = meta["c0"]; lam = meta["lambda_"]
    nsims = meta["nsims"]; ncoordsout = meta["ncoordsout"]
    outc = np.asarray(xdc["outcoords"]); npx = int(np.asarray(xdc["outcoords2"]).shape[0])
    lab = outc[:, 3].astype(int)                            # sub-element -> element label (1..npx)

    # --- per-transmit RF, average sub-elements -> (n_tx, nframes, npx) ---
    # channel_data (nsims,nframes,npx), if given, is used as-is (already filtered);
    # otherwise read genout, average sub-elements per element, and bandpass.
    t0 = np.zeros(nsims)
    if channel_data is not None:
        px = np.asarray(channel_data, np.float64)
        for i in range(nsims):
            t0[i] = np.argmax(np.abs(hilbert(px[i, :, i])))
    else:
        px = np.zeros((nsims, meta["nframes"], npx), np.float64)
        for i in range(nsims):
            g = io_dat.read_genout(f"{rundir}/{io_dat.genout_name(i)}", ncoordsout)  # (nframes,ncoordsout)
            for tt in range(1, npx + 1):
                sel = lab == tt
                if sel.any(): px[i, :, tt - 1] = g[:, sel].mean(1)
            env = np.abs(hilbert(px[i, :, i]))              # element i self-receive
            t0[i] = np.argmax(env)
        for i in range(nsims):
            px[i] = _bandpass(px[i], fs, f0, bw)

    # --- geometry: element centers + sector grid ---
    o2 = np.asarray(xdc["outcoords2"], float); i2 = np.asarray(xdc["incoords2"], float)
    rx = np.stack([(o2[:, 0] - o2[:, 0].mean()) * dY, np.zeros(npx), o2[:, 1] * dY], 1)
    tx = np.stack([(i2[:, 0] - i2[:, 0].mean()) * dY, np.zeros(npx), i2[:, 1] * dY], 1)
    rad = float(np.asarray(xdc["rad"])); cen = np.asarray(xdc["cen"], float).ravel()
    cenz = cen[1] * dY
    nY = meta["nY"]                                         # INTERIOR nY
    offset = (rad + cen[1]) * dY
    rs = np.arange(5e-3, nY * dY - offset, lam / 4) + rad * dY
    angles = np.linspace(-0.30, 0.30, 96)
    AA, RR = np.meshgrid(angles, rs)                        # (R, A)
    scan = np.stack([(RR * np.sin(AA)).ravel(), np.zeros(RR.size),
                     (RR * np.cos(AA) + cenz).ravel()], 1).astype(np.float32)

    # --- tx wavefront arrivals: spherical from each transmit element ---
    arrivals = np.linalg.norm(tx[:, None, :] - scan[None, :, :], axis=2) / c0   # (n_tx, n_scan)

    # optional RTBF-style transmit acceptance cone (boxcar). MACH has no per-(tx,pixel)
    # mask, so exclude out-of-cone pairs by pushing their arrival far past the record
    # length -> the interpolator flags the sample invalid -> zero contribution.
    # f-number is along each pixel's STEERED beam axis (sector): accept transmit n for
    # pixel p iff perp_offset / axial_dist <= 1/(2*txfnum), where the beam axis points
    # from the pixel back toward the array, dir = -(sin phi, cos phi).
    tx_inside = None                                           # (n_tx, n_scan) bool, for mean-norm
    if txfnum:
        phi = AA.ravel()                                       # (n_scan,) apex angle per pixel
        ax_x, ax_z = -np.sin(phi), -np.cos(phi)               # beam dir toward array
        vx = tx[:, None, 0] - scan[None, :, 0]                # (n_tx, n_scan)
        vz = tx[:, None, 2] - scan[None, :, 2]
        axial = vx * ax_x[None, :] + vz * ax_z[None, :]       # along-beam distance to element
        perp = np.sqrt(np.maximum(vx*vx + vz*vz - axial*axial, 0.0))
        outside = (axial <= 0) | (perp > axial / (2.0 * txfnum))
        tx_inside = ~outside
        arrivals = np.where(outside, 1e3, arrivals)            # 1e3 s >> record -> masked

    # --- form complex channel data per the chosen convention ---
    if convention == "iq":
        # baseband IQ: demodulate (fs/f0~4.4 samples/cycle is coarse for RF interp),
        # MACH re-imposes the carrier via modulation_freq_hz.
        tsamp = np.arange(px.shape[1]) / fs
        cdat = hilbert(px, axis=1) * np.exp(-2j * np.pi * f0 * tsamp)[None, :, None]
        modfreq = mod_freq_sign * f0
    elif convention == "analytic_rf":
        # analytic RF (RTBF convention): keep the carrier, no demod; MACH does pure
        # delay-and-sum of the oscillating analytic samples (modulation_freq_hz=0).
        cdat = hilbert(px, axis=1)
        modfreq = 0.0
    else:
        raise ValueError(f"convention must be 'iq' or 'analytic_rf', got {convention!r}")
    cd = np.transpose(cdat, (0, 2, 1))[..., None].astype(np.complex64)  # (n_tx, npx, nframes, 1)

    # t0 is in FRAME units (argmax of the modT-downsampled genout); RTBF uses
    # t_start = t0*dT with the FINE solver dT (NOT the frame period dT*modT), so
    # no modT factor here -- a *modT over-shifts the image and caps RTBF corr.
    rx_start_s = -float(np.median(t0)) * dT                           # sample0 vs transmit
    img = _mach_compound(cd, rx.astype(np.float32), scan, arrivals.astype(np.float32),
                         rx_start_s=rx_start_s, fs=fs, fnum=fnum, c0=c0,
                         modfreq=modfreq, interp=interp, tukey_alpha=tukey_alpha)
    img = np.asarray(img).reshape(len(rs) * len(angles))             # (n_scan,)

    if normalize == "mean":
        # RTBF returns sum/sumcount; MACH returns the raw sum. Divide each pixel by its
        # contributing (tx * rx) count to match -- removes the depth-growing aperture gain.
        rxc = (np.abs(rx[None, :, 0] - scan[:, None, 0])         # MACH aperture: |x_rx-x_px|
               <= scan[:, None, 2] / (2.0 * fnum)).sum(1)        # <= z_px/(2*fnum)  -> (n_scan,)
        txc = tx_inside.sum(0) if tx_inside is not None else float(nsims)
        img = img / np.maximum(rxc * txc, 1.0)
    elif normalize != "sum":
        raise ValueError(f"normalize must be 'sum' or 'mean', got {normalize!r}")
    img = img.reshape(len(rs), len(angles))                          # (R, A)
    env = np.abs(img); env = env / env.max()
    bmode_db = 20 * np.log10(env + 1e-12)

    beam_orig = rad * dX * np.stack([np.sin(angles), np.cos(angles)], 1) + np.array([0, cenz])
    b_sector, x, z = scan_convert(bmode_db, rs - rad * dY, angles, beam_orig, [0, cenz], lam / 8)
    return dict(b_sector=b_sector, x=x, z=z, env_polar=env, bmode_db=bmode_db,
                angles=angles, rs=rs - rad * dY, t0=t0)
