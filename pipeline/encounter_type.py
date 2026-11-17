"""
encounter_type.py - the type of an encounter between two road users, decided the way published tools decide it.

Used by safety.py; not run on its own.

SUMO's SSM device sorts encounters by the lanes used before and after the conflict point: lead/follow (same lanes before),
merging (different lanes before, the same after), crossing (different before and after), and it computes no
post-encroachment time for lead/follow. FHWA's SSAM calls a conflict rear end under 30 degrees, crossing over 80 degrees,
lane change between, and lets the lanes overrule the angle when it has them.

Recorded tracks carry no lane number, so the lane is read from the paths themselves:
  following   the second road user approached the place along the path the first one had driven: within 3 m of it (one
              lane; these roads have one lane each way, so side by side in the same direction is the same lane, the bike
              lane or the kerb) and within 30 degrees of its direction. What the first one does afterwards (turns off,
              pulls in) does not change that.
  merging     they approached on different paths and the second one leaves along the path of the first, or the two point
              the same way (under 30 degrees) at the place.
  oncoming    opposite directions, and the second road user's centre path does not cross the first one's within 10 m of
              the place in the 10 s after the first has passed there. Two vehicles that pass in opposite lanes share a
              cell only because their bodies both reach the centre line; a left turn behind an oncoming vehicle does
              cross its path, and is a crossing.
  crossing    everything else: paths from different directions that cross and separate again.
Post-encroachment time is counted for merging and crossing only. Following pairs belong to time-to-collision."""
import math
import numpy as np

LANE = 3.0              # m: the second road user is in the first one's lane (or right beside it) within this
SAME_DIR = 30.0         # degrees between local directions of travel (SSAM's rear-end angle)
EXTEND = 15.0           # m: a track that begins or ends inside the scene is continued straight this far
CROSS_NEAR_M, CROSS_SOON_S = 10.0, 10.0   # opposite directions: the paths cross this near the place, the second this soon after the first
REACH, SKIP = 10.0, 2.0 # the stretch compared: 10 m of travel, leaving out the 2 m at the place itself
MIN_SPAN = 4.0          # m of that stretch that must exist
SHARE = 0.6             # share of the stretch that must agree


def thin(P, T, step=0.5):
    if len(P) == 0: return P, T
    keep = [0]
    for i in range(1, len(P)):
        if np.hypot(*(P[i] - P[keep[-1]])) >= step: keep.append(i)
    return P[keep], T[keep]


def directions(P):
    """unit direction of travel at each point of a thinned path"""
    if len(P) < 2: return np.zeros((len(P), 2))
    d = np.zeros_like(P); d[1:-1] = P[2:] - P[:-2]; d[0] = P[1] - P[0]; d[-1] = P[-1] - P[-2]
    n = np.hypot(d[:, 0], d[:, 1]); n[n == 0] = 1.0
    return d / n[:, None]


def stretch(o, t, sign):
    """the REACH metres of travel before (sign -1) or after (sign +1) time t, without the SKIP metres at t"""
    P = np.c_[o["E"], o["N"]]; T = o["T"]
    i = min(int(np.searchsorted(T, t)), len(T) - 1)
    idx = range(i, -1, -1) if sign < 0 else range(i, len(T))
    pts, d, last = [], 0.0, P[i]
    for k in idx:
        d += float(np.hypot(*(P[k] - last))); last = P[k]
        if d > REACH + SKIP: break
        if d >= SKIP: pts.append((d, P[k][0], P[k][1], T[k]))
    if len(pts) < 2: return None
    A = np.array(pts)
    if A[-1, 0] - A[0, 0] < MIN_SPAN: return None
    if sign < 0: A = A[::-1]                                         # in order of travel
    Q, Tq = thin(A[:, 1:3], A[:, 3])
    return Q if len(Q) >= 3 else None


def reference(o, t, sign, far=40.0):
    """the first road user's path over the `far` metres of travel before (sign -1) or after (sign +1) time t, however long
    it stood on the way, plus one second on the other side of t. Where its track begins or ends short of that, the path is
    carried on straight for EXTEND metres along its first or last direction of travel: a vehicle first tracked inside the
    junction did arrive along the lane behind it."""
    P = np.c_[o["E"], o["N"]]; T = o["T"]
    i = min(int(np.searchsorted(T, t)), len(T) - 1)
    step = np.r_[0.0, np.hypot(np.diff(P[:, 0]), np.diff(P[:, 1]))]; cum = np.cumsum(step)
    if sign < 0:
        j0 = int(np.searchsorted(cum, cum[i] - far)); j1 = min(int(np.searchsorted(T, t + 1.0)), len(T) - 1)
        short_start, short_end = (j0 == 0 and cum[i] - cum[0] < far), False
    else:
        j0 = max(int(np.searchsorted(T, t - 1.0)), 0); j1 = min(int(np.searchsorted(cum, cum[i] + far)), len(T) - 1)
        short_start, short_end = False, (j1 == len(T) - 1 and cum[-1] - cum[i] < far)
    Q, Tq = thin(P[j0:j1 + 1], T[j0:j1 + 1])
    if len(Q) < 3: return Q
    def lead(R):                                                     # direction over the first 3 m of travel
        for k in range(1, len(R)):
            if np.hypot(*(R[k] - R[0])) >= 3.0: return (R[k] - R[0]) / np.hypot(*(R[k] - R[0]))
        return None
    if short_start:
        u = lead(Q)
        if u is not None: Q = np.vstack([[Q[0] - u * d for d in np.arange(EXTEND, 0, -0.5)], Q])
    if short_end:
        u = lead(Q[::-1])
        if u is not None: Q = np.vstack([Q, [Q[-1] - u * d for d in np.arange(0.5, EXTEND + 0.01, 0.5)]])
    return Q


def window(o, t0, t1):
    m = (o["T"] >= t0) & (o["T"] <= t1)
    return thin(np.c_[o["E"][m], o["N"][m]], o["T"][m])[0]


def agree(Q, R):
    """share of the points of stretch Q that lie along reference path R (within LANE, same direction); None if R does not
    reach alongside Q"""
    if Q is None or len(R) < 3: return None
    dq, dr = directions(Q), directions(R)
    same = cov = 0
    for p, u in zip(Q, dq):
        dist = np.hypot(R[:, 0] - p[0], R[:, 1] - p[1]); j = int(np.argmin(dist))
        if j == 0 and float((p - R[0]) @ dr[0]) < -1.0: continue     # the point lies before the reference path begins
        if j == len(R) - 1 and float((p - R[-1]) @ dr[-1]) > 1.0: continue   # or after it ends
        cov += 1
        ang = math.degrees(math.acos(max(-1.0, min(1.0, float(u @ dr[j])))))
        if ang <= SAME_DIR and dist[j] <= LANE: same += 1
    if cov < 3: return None
    return same / cov


def crosses_soon(A, B, t, near):
    """the centre path of B crosses that of A within CROSS_NEAR_M of the place, B passing the crossing point up to
    CROSS_SOON_S after A did"""
    ma = (A["T"] >= t - 20.0) & (A["T"] <= t + 20.0); mb = (B["T"] >= t - 20.0) & (B["T"] <= t + 20.0)
    P, TP = thin(np.c_[A["E"][ma], A["N"][ma]], A["T"][ma]); Q, TQ = thin(np.c_[B["E"][mb], B["N"][mb]], B["T"][mb])
    if len(P) < 2 or len(Q) < 2: return False
    a, r = P[:-1], P[1:] - P[:-1]; c, s_ = Q[:-1], Q[1:] - Q[:-1]
    for i in range(len(a)):
        den = r[i, 0] * s_[:, 1] - r[i, 1] * s_[:, 0]; qp = c - a[i]
        with np.errstate(divide="ignore", invalid="ignore"):
            tt = (qp[:, 0] * s_[:, 1] - qp[:, 1] * s_[:, 0]) / den; u = (qp[:, 0] * r[i, 1] - qp[:, 1] * r[i, 0]) / den
        for j in np.nonzero((np.abs(den) > 1e-9) & (tt >= 0) & (tt <= 1) & (u >= 0) & (u <= 1))[0]:
            X = a[i] + tt[j] * r[i]; dt = float(TQ[j] - TP[i])
            if np.hypot(X[0] - near[0], X[1] - near[1]) <= CROSS_NEAR_M and 0.0 <= dt <= CROSS_SOON_S: return True
    return False


def event_angle(A, B, tA, tB):
    ia = min(int(np.searchsorted(A["T"], tA)), len(A["T"]) - 1); ib = min(int(np.searchsorted(B["T"], tB)), len(B["T"]) - 1)
    va, vb = A["V"][ia], B["V"][ib]; na, nb = np.hypot(*va), np.hypot(*vb)
    if na <= 0.3 or nb <= 0.3: return None
    return math.degrees(math.acos(max(-1.0, min(1.0, float(va @ vb) / (na * nb)))))


def ssam_type(ang):
    """SSAM's angle rule (no lane information): rear end under 30 degrees, crossing over 80, lane change between"""
    if ang is None: return "none"
    return "rear end" if ang < 30 else "crossing" if ang > 80 else "lane change"


def encounter(objs, gid_a, gid_b, t_sec, seconds, E, N):
    """type, and what it rests on, for the encounter of gid_a (passes first, leaves at t_sec - seconds) and gid_b (arrives at
    t_sec) at the place (E, N)"""
    A, B = objs.get(int(gid_a)), objs.get(int(gid_b))
    if A is None or B is None: return dict(type="crossing", why="no track", ang=None)
    tA, tB = t_sec - seconds, t_sec
    ang = event_angle(A, B, tA, tB)
    if ang is not None and ang > 150 and not crosses_soon(A, B, t_sec, (E, N)):
        return dict(type="oncoming", why="opposite directions, paths do not cross", ang=ang)
    before = agree(stretch(B, tB, -1), reference(A, tA, -1))
    after = agree(stretch(B, tB, +1), reference(A, tA, +1))
    out = dict(ang=ang, before=before, after=after)
    if before is not None and before >= SHARE: return dict(out, type="following", why="same lane before")
    if before is None and ang is not None and ang < 30:              # no approach to compare: SSAM's angle rule
        return dict(out, type="following", why="no approach on record, angle under 30")
    if after is not None and after >= SHARE: return dict(out, type="merging", why="different paths before, same lane after")
    if before is not None and ang is not None and ang < 30:          # came another way and now point the same way
        return dict(out, type="merging", why="different paths before, same direction at the place")
    if before is None: return dict(out, type="crossing", why="no approach on record, angle 30 or more or none")
    return dict(out, type="crossing", why="different paths")
