#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3D Container Packet Packing via Local Search
============================================

Packs rectangular packets into a rectangular container so that the total
value of the packets placed inside is maximised, using one of three
local-search algorithms:

    T = 1   Steepest Ascent Hill Climbing (strict improvement)
    T = 2   Simulated Annealing (geometric cooling)
    T = 3   Genetic Algorithm (roulette selection, uniform crossover + repair)

Usage
-----
    python packet_packer.py <input_file> [--outdir DIR] [--seed N] [--no-show]

Input file format
-----------------
    L H W M                 container dimensions + maximum weight
    T [Op1 Op2]             algorithm type; Op1/Op2 = GA starting population
                            and iteration (generation) count; ignored for T!=3
    N                       number of packets
    id l h w ol oh ow v we f     (one line per packet)
        id          unique identifier (string, no spaces)
        l h w       length / height / width of the packet
        ol oh ow    orientation: which axis (x, y or z) the length/height/width
                    are parallel to, e.g. "x y z" (6 permutations possible)
        v           value of the packet
        we          weight of the packet
        f           fragile flag (1 = fragile, 0 = not fragile)

Blank lines and lines starting with '#' are ignored.
"""

import argparse
import itertools
import math
import os
import random
import sys
import time

import matplotlib

# ============================================================================
# Constants
# ============================================================================

OUTSIDE_MARGIN = 1          # parking offset beyond container upper corner

# --- Steepest Ascent Hill Climbing ------------------------------------------
HC_MAX_CANDIDATES = 6000    # safety cap on enumerated target positions/packet

# --- Simulated Annealing (fixed constants) -----------------------------------
SA_T0 = 1000.0              # initial temperature
SA_ALPHA = 0.995            # geometric cooling factor: T *= alpha
SA_MAX_ITER = 15000         # maximum iterations

# --- Genetic Algorithm --------------------------------------------------------
GA_DEFAULT_POP = 30         # starting population used when Op1 is missing
GA_DEFAULT_GEN = 100        # generation count used when Op2 is missing
GA_ELITE = 2                # individuals carried over unchanged each generation
GA_MUTATION_PROB = 0.25     # probability that a child gets mutated
GA_MUTATION_TRIES = 10      # random moves attempted per mutation
GA_REPAIR_TRIES = 40        # random re-placement attempts during crossover repair

# --- shared -------------------------------------------------------------------
RANDOM_INSERT_TRIES = 40    # insertion attempts when building a random state
MOVE_P = 0.50               # probability that a random move is a "move"
ROTATE_P = 0.22             # probability that a random move is a "rotation"
# (remaining probability = swap)

AXES = ('x', 'y', 'z')
ORIENTATIONS = tuple(sorted(itertools.permutations('xyz')))


class InputError(Exception):
    """Raised when the input file cannot be parsed."""


# ============================================================================
# Problem model
# ============================================================================

class Packet:
    __slots__ = ('pid', 'l', 'h', 'w', 'value', 'weight', 'fragile', 'base_orient')

    def __init__(self, pid, l, h, w, value, weight, fragile, base_orient):
        self.pid = pid
        self.l, self.h, self.w = l, h, w
        self.value = value
        self.weight = weight
        self.fragile = fragile
        self.base_orient = base_orient

    @property
    def dims(self):
        return (self.l, self.h, self.w)


class Problem:
    def __init__(self, L, H, W, M, packets):
        self.L, self.H, self.W, self.M = L, H, W, M
        self.packets = packets
        self.parking = (L + OUTSIDE_MARGIN, H + OUTSIDE_MARGIN, W + OUTSIDE_MARGIN)


def oriented_dims(dims, orient):
    """Effective (dx, dy, dz) of a packet with (l, h, w) under `orient`.

    `orient` = (axis of length, axis of height, axis of width).
    """
    m = {orient[0]: dims[0], orient[1]: dims[1], orient[2]: dims[2]}
    return (m['x'], m['y'], m['z'])


def rotated(orient, axis):
    """Orientation after a 90-degree rotation about the given world axis.

    Rotating about x swaps the y/z extents (height <-> width),
    about y swaps the x/z extents (length <-> width),
    about z swaps the x/y extents (length <-> height).
    """
    a, b = {'x': (1, 2), 'y': (0, 2), 'z': (0, 1)}[axis]
    o = list(orient)
    o[a], o[b] = o[b], o[a]
    return tuple(o)


def parse_input(path):
    """Parse the input file -> (Problem, T, Op1, Op2)."""
    try:
        with open(path, 'r', encoding='utf-8') as fh:
            raw = fh.read()
    except OSError as exc:
        raise InputError('cannot read input file: %s' % exc)

    lines = []
    for line in raw.splitlines():
        s = line.strip()
        if not s or s.startswith('#'):
            continue
        lines.append(s.split())

    if len(lines) < 3:
        raise InputError('input must contain at least 3 header lines '
                         '(L H W M / T [Op1 Op2] / N)')

    def to_int(tokens, what):
        out = []
        for t in tokens:
            try:
                out.append(int(t))
            except ValueError:
                raise InputError('non-integer value %r on the %s line' % (t, what))
        return out

    dims_line = to_int(lines[0], 'container')
    if len(dims_line) != 4:
        raise InputError('container line must have exactly 4 values: L H W M')
    L, H, W, M = dims_line
    if L <= 0 or H <= 0 or W <= 0:
        raise InputError('container dimensions must be positive')
    if M < 0:
        raise InputError('maximum weight M must be non-negative')

    tline = lines[1]
    if len(tline) < 1 or len(tline) > 3:
        raise InputError('algorithm line must be: T [Op1 Op2]')
    T = to_int(tline[:1], 'algorithm')[0]
    if T not in (1, 2, 3):
        raise InputError('algorithm type must be 1 (SAHC), 2 (SA) or 3 (GA), got %d' % T)
    op1 = op2 = None
    if len(tline) >= 2:
        try:
            op1 = int(float(tline[1]))
        except ValueError:
            raise InputError('invalid Op1 value: %r' % tline[1])
    if len(tline) >= 3:
        try:
            op2 = int(float(tline[2]))
        except ValueError:
            raise InputError('invalid Op2 value: %r' % tline[2])

    n_line = to_int(lines[2], 'N')
    if len(n_line) != 1:
        raise InputError('N line must contain exactly one value')
    n_packets = n_line[0]
    if n_packets <= 0:
        raise InputError('N must be positive')
    if len(lines) < 3 + n_packets:
        raise InputError('expected %d packet lines, found %d'
                         % (n_packets, len(lines) - 3))
    if len(lines) > 3 + n_packets:
        print('warning: ignoring %d extra line(s) after the packet list'
              % (len(lines) - 3 - n_packets), file=sys.stderr)

    packets = []
    seen = set()
    for k in range(n_packets):
        tok = lines[3 + k]
        if len(tok) != 10:
            raise InputError('packet line %d must have 10 fields, got %d: %s'
                             % (k + 1, len(tok), ' '.join(tok)))
        pid = tok[0]
        if pid in seen:
            raise InputError('duplicate packet id: %s' % pid)
        seen.add(pid)
        try:
            l = int(tok[1]); h = int(tok[2]); w = int(tok[3])
            v = int(tok[7]); we = int(tok[8]); f = int(tok[9])
        except ValueError:
            raise InputError('packet %s: dimensions, value, weight and flag '
                             'must be integers' % pid)
        orient = tuple(a.lower() for a in tok[4:7])
        if orient not in ORIENTATIONS:
            raise InputError('packet %s: orientation (%s) must be a permutation '
                             'of x/y/z' % (pid, ' '.join(tok[4:7])))
        if l <= 0 or h <= 0 or w <= 0:
            raise InputError('packet %s: dimensions must be positive' % pid)
        if v < 0 or we < 0:
            raise InputError('packet %s: value and weight must be non-negative' % pid)
        if f not in (0, 1):
            raise InputError('packet %s: fragile flag must be 0 or 1' % pid)
        packets.append(Packet(pid, l, h, w, v, we, bool(f), orient))

    return Problem(L, H, W, M, packets), T, op1, op2


# ============================================================================
# Geometry helpers
# ============================================================================

def _vol_overlap(a, b):
    """True if two boxes (x0, y0, z0, x1, y1, z1) overlap with positive volume."""
    return (a[0] < b[3] and b[0] < a[3] and
            a[1] < b[4] and b[1] < a[4] and
            a[2] < b[5] and b[2] < a[5])


def _xz_touch(a, b):
    """True if two boxes touch/overlap when projected on the x-z plane
    (closed rectangles: edge or corner contact counts)."""
    return (a[0] <= b[3] and b[0] <= a[3] and
            a[2] <= b[5] and b[2] <= a[5])


# ============================================================================
# State and validity checking
# ============================================================================

class State:
    """inside/outside flag, a position(minimum corner) and an orientation.
    Outside packets sit at the parking spot (L+1, H+1, W+1)."""

    __slots__ = ('problem', 'inside', 'pos', 'orient', 'total_value', 'total_weight')

    def __init__(self, problem):
        self.problem = problem
        n = len(problem.packets)
        self.inside = [False] * n
        self.pos = [problem.parking] * n
        self.orient = [p.base_orient for p in problem.packets]
        self.total_value = 0
        self.total_weight = 0

    def copy(self):
        s = State.__new__(State)
        s.problem = self.problem
        s.inside = list(self.inside)
        s.pos = list(self.pos)
        s.orient = list(self.orient)
        s.total_value = self.total_value
        s.total_weight = self.total_weight
        return s

    def dims(self, i):
        return oriented_dims(self.problem.packets[i].dims, self.orient[i])

    def box(self, i):
        x, y, z = self.pos[i]
        dx, dy, dz = self.dims(i)
        return (x, y, z, x + dx, y + dy, z + dz)

    def _inside_indices(self):
        return [j for j, f in enumerate(self.inside) if f]

    # --- individual rule checks ---------------------------------------------

    def _bounds_ok(self, i):
        L, H, W = self.problem.L, self.problem.H, self.problem.W
        b = self.box(i)
        return (0 <= b[0] and b[3] <= L and
                0 <= b[1] and b[4] <= H and
                0 <= b[2] and b[5] <= W)

    def _has_support(self, i):
        """Base touches the container floor or the top of another inside packet
        (at least one point of contact)."""
        b = self.box(i)
        if b[1] == 0:
            return True
        for j in self._inside_indices():
            if j == i:
                continue
            o = self.box(j)
            if o[4] == b[1] and _xz_touch(b, o):
                return True
        return False

    def _pair_ok(self, i, j):
        """All pairwise rules between two inside packets."""
        a = self.box(i)
        b = self.box(j)
        if _vol_overlap(a, b):
            return False
        pi = self.problem.packets[i]
        pj = self.problem.packets[j]
        if pi.fragile and b[1] == a[4] and _xz_touch(a, b):
            return False            # j rests on top of fragile i
        if pj.fragile and a[1] == b[4] and _xz_touch(a, b):
            return False            # i rests on top of fragile j
        return True

    # --- validity ------------------------------------------------------------

    def validate_incremental(self, changed, old_ymaxs):
        """Validate the state after changing only packets in `changed`.

        `old_ymaxs` = set of previous top heights of the changed packets, used
        to re-check the support of unchanged packets that may have rested on
        them. The state must have been valid before the change.
        """
        if self.total_weight > self.problem.M:
            return False
        ins = self._inside_indices()
        inset = set(ins)
        # bounds of the changed packets
        for c in changed:
            if c in inset and not self._bounds_ok(c):
                return False
        # pairwise rules: changed vs every other inside packet
        for c in changed:
            if c not in inset:
                continue
            for j in ins:
                if j != c and not self._pair_ok(c, j):
                    return False
        # support of changed packets
        for c in changed:
            if c in inset and not self._has_support(c):
                return False
        # support of unchanged packets that rested at a former top height
        if old_ymaxs:
            for j in ins:
                if j not in changed and self.pos[j][1] in old_ymaxs:
                    if not self._has_support(j):
                        return False
        return True

    def validate_full(self):
        """Complete validity check (used as a safety net on initial/final states)."""
        ins = self._inside_indices()
        for i in ins:
            if not self._bounds_ok(i):
                return False
            if not self._has_support(i):
                return False
        for a in range(len(ins)):
            for b in range(a + 1, len(ins)):
                if not self._pair_ok(ins[a], ins[b]):
                    return False
        return self.total_weight <= self.problem.M


# ============================================================================
# Transitions with snapshot / rollback
# ============================================================================

def _ymax_of(state, i):
    """Current top height of packet i (None if outside)."""
    if not state.inside[i]:
        return None
    return state.pos[i][1] + state.dims(i)[1]


def _capture(state, idxs):
    return ([(i, state.inside[i], state.pos[i], state.orient[i]) for i in idxs],
            state.total_value, state.total_weight)


def _restore(state, snap):
    entries, tv, tw = snap
    for (i, inside, pos, orient) in entries:
        state.inside[i] = inside
        state.pos[i] = pos
        state.orient[i] = orient
    state.total_value = tv
    state.total_weight = tw


def _assign(state, i, inside, pos, orient):
    """Write a placement for packet i, keeping the cached totals in sync."""
    p = state.problem.packets[i]
    was = state.inside[i]
    if was and not inside:
        state.total_value -= p.value
        state.total_weight -= p.weight
    elif inside and not was:
        state.total_value += p.value
        state.total_weight += p.weight
    state.inside[i] = inside
    state.pos[i] = pos
    state.orient[i] = orient


def try_place(state, i, inside, pos, orient):
    """Attempt to change packet i's placement.

    Returns a snapshot if the resulting state is valid (change kept), else
    None and the state is restored. `inside=False` packets are parked at
    (L+1, H+1, W+1) automatically.
    """
    if (inside == state.inside[i] and pos == state.pos[i]
            and orient == state.orient[i]):
        return None                                     # no-op
    old_ym = _ymax_of(state, i)
    snap = _capture(state, [i])
    _assign(state, i, inside, pos, orient)
    suspects = {old_ym} if old_ym is not None else set()
    if state.validate_incremental((i,), suspects):
        return snap
    _restore(state, snap)
    return None


def try_swap(state, i, j):
    """Swap the placements (inside flag + position) of packets i and j.

    Each packet keeps its own orientation. Returns a snapshot if valid, else
    None with the state restored. Swapping two outside packets is a no-op.
    """
    if state.inside[i] == state.inside[j] and state.pos[i] == state.pos[j]:
        return None                                     # both outside (or identical)
    old_ys = {_ymax_of(state, i), _ymax_of(state, j)}
    old_ys.discard(None)
    snap = _capture(state, [i, j])
    pi = (state.inside[i], state.pos[i])
    pj = (state.inside[j], state.pos[j])
    _assign(state, i, pj[0], pj[1], state.orient[i])
    _assign(state, j, pi[0], pi[1], state.orient[j])
    if state.validate_incremental((i, j), old_ys):
        return snap
    _restore(state, snap)
    return None


# ============================================================================
# Candidate target positions
# ============================================================================

def axis_sets(state, i, orient):
    """Canonical candidate coordinate sets for placing packet i (with the
    given orientation) without overlapping the packets already inside:
        x in {0} u {q.x0 - dx, q.x1},  z in {0} u {q.z0 - dz, q.z1},
        y in {0} u {q.y1}   (y must rest on the floor or on a top face)
    Also returns a map top-height -> boxes of inside packets, for O(1)-ish
    support pre-checks."""
    dx, dy, dz = oriented_dims(state.problem.packets[i].dims, orient)
    L, H, W = state.problem.L, state.problem.H, state.problem.W
    xs = {0}
    ys = {0}
    zs = {0}
    sup = {}
    for j in state._inside_indices():
        if j == i:
            continue
        b = state.box(j)
        xs.add(b[0] - dx)
        xs.add(b[3])
        ys.add(b[4])
        zs.add(b[2] - dz)
        zs.add(b[5])
        sup.setdefault(b[4], []).append(b)
    xs = sorted(v for v in xs if 0 <= v <= L - dx)
    ys = sorted(v for v in ys if 0 <= v <= H - dy)
    zs = sorted(v for v in zs if 0 <= v <= W - dz)
    return xs, ys, zs, sup


def _support_ok_at(x, y, z, dx, dz, sup):
    """Cheap pre-check: would a packet with footprint dx*dz at (x, y, z) have
    support? (full validation still happens in try_place)"""
    if y == 0:
        return True
    for b in sup.get(y, ()):
        if x <= b[3] and b[0] <= x + dx and z <= b[5] and b[2] <= z + dz:
            return True
    return False


def candidate_positions(state, i, rng, orient, cap):
    """All canonical candidate positions for packet i (support pre-checked)."""
    xs, ys, zs, sup = axis_sets(state, i, orient)
    dx, _dy, dz = oriented_dims(state.problem.packets[i].dims, orient)
    cur = state.pos[i] if state.inside[i] else None
    out = []
    for x in xs:
        for z in zs:
            for y in ys:
                if cur is not None and x == cur[0] and y == cur[1] and z == cur[2]:
                    continue
                if _support_ok_at(x, y, z, dx, dz, sup):
                    out.append((x, y, z))
    if cap and len(out) > cap:
        out = rng.sample(out, cap)
    return out


def random_candidate(state, i, rng, orient, tries=8):
    """One random support-feasible candidate position for packet i, or None."""
    xs, ys, zs, sup = axis_sets(state, i, orient)
    if not xs or not ys or not zs:
        return None
    dx, _dy, dz = oriented_dims(state.problem.packets[i].dims, orient)
    cur = state.pos[i] if state.inside[i] else None
    for _ in range(tries):
        x = rng.choice(xs)
        y = rng.choice(ys)
        z = rng.choice(zs)
        if cur is not None and x == cur[0] and y == cur[1] and z == cur[2]:
            continue
        if _support_ok_at(x, y, z, dx, dz, sup):
            return (x, y, z)
    return None


# ============================================================================
# Random valid transition (used by SA and by GA mutation)
# ============================================================================

def random_neighbor(state, rng):
    """Attempt one random state transition (move / rotate / swap).

    Returns (applied, snapshot). If applied, the state now holds the
    neighbouring state and the caller may accept it or roll back with the
    snapshot. If the proposal was invalid, the state is untouched.
    """
    n = len(state.problem.packets)
    r = rng.random()
    if r < MOVE_P:                                       # --- move ---
        i = rng.randrange(n)
        if state.inside[i] and rng.random() < 0.15:
            snap = try_place(state, i, False, state.problem.parking, state.orient[i])
            return (snap is not None), snap
        cand = random_candidate(state, i, rng, state.orient[i])
        if cand is None:
            return False, None
        snap = try_place(state, i, True, cand, state.orient[i])
        return (snap is not None), snap
    elif r < MOVE_P + ROTATE_P:                          # --- rotate ---
        ins = state._inside_indices()
        if not ins:
            return False, None
        i = rng.choice(ins)
        axis = rng.choice(AXES)
        snap = try_place(state, i, True, state.pos[i], rotated(state.orient[i], axis))
        return (snap is not None), snap
    else:                                                # --- swap ---
        if n < 2:
            return False, None
        i = rng.randrange(n)
        j = rng.randrange(n)
        while j == i:
            j = rng.randrange(n)
        snap = try_swap(state, i, j)
        return (snap is not None), snap


def random_valid_state(problem, rng):
    """Randomised but valid initial state: packets are inserted in random
    order at random candidate positions with random orientations; packets
    that cannot be placed stay outside."""
    state = State(problem)
    order = list(range(len(problem.packets)))
    rng.shuffle(order)
    for i in order:
        for _ in range(RANDOM_INSERT_TRIES):
            o = rng.choice(ORIENTATIONS)
            cand = random_candidate(state, i, rng, o)
            if cand is not None and try_place(state, i, True, cand, o) is not None:
                break
    if not state.validate_full():
        raise RuntimeError('random initial state failed the validity check')
    return state


# ============================================================================
# Algorithm 1: Steepest Ascent Hill Climbing
# ============================================================================

def run_hc(state, rng):
    """Steepest Ascent Hill Climbing."""
    history = [state.total_value]
    iterations = 0
    n = len(state.problem.packets)
    while True:
        best_delta = 0
        best_move = None        # ('place', i, inside, pos, orient) | ('swap', i, j)
        for i in range(n):
            o = state.orient[i]
            if state.inside[i]:
                for axis in AXES:                        # rotations
                    o2 = rotated(o, axis)
                    snap = try_place(state, i, True, state.pos[i], o2)
                    if snap is not None:
                        delta = state.total_value - snap[1]
                        _restore(state, snap)
                        if delta > best_delta:
                            best_delta, best_move = delta, ('place', i, True, state.pos[i], o2)
                snap = try_place(state, i, False, state.problem.parking, o)
                if snap is not None:                     # move outside
                    delta = state.total_value - snap[1]
                    _restore(state, snap)
                    if delta > best_delta:
                        best_delta, best_move = delta, ('place', i, False,
                                                        state.problem.parking, o)
            for pos in candidate_positions(state, i, rng, o, HC_MAX_CANDIDATES):
                snap = try_place(state, i, True, pos, o)
                if snap is not None:
                    delta = state.total_value - snap[1]
                    _restore(state, snap)
                    if delta > best_delta:
                        best_delta, best_move = delta, ('place', i, True, pos, o)
        for i in range(n):                               # swaps
            for j in range(i + 1, n):
                snap = try_swap(state, i, j)
                if snap is not None:
                    delta = state.total_value - snap[1]
                    _restore(state, snap)
                    if delta > best_delta:
                        best_delta, best_move = delta, ('swap', i, j)
        if best_move is None:
            break                                        # local optimum
        if best_move[0] == 'place':
            if try_place(state, *best_move[1:]) is None:
                raise RuntimeError('HC re-apply failed (internal error)')
        else:
            if try_swap(state, best_move[1], best_move[2]) is None:
                raise RuntimeError('HC re-apply failed (internal error)')
        iterations += 1
        history.append(state.total_value)
    return {'final_state': state, 'history': history,
            'iterations': iterations, 'final_iter': iterations}


# ============================================================================
# Algorithm 2: Simulated Annealing (geometric cooling)
# ============================================================================

def run_sa(state, rng):
    """Simulated Annealing with geometric cooling T <- alpha * T.
    The reported final state is the best state visited."""
    temperature = SA_T0
    stayed = 0
    best = state.copy()
    best_value = state.total_value
    best_iter = 0
    best_hist = [best_value]
    cur_hist = [state.total_value]
    accept_hist = []            # e^(dE/T) per iteration (NaN for invalid proposals)
    it = 0
    while it < SA_MAX_ITER:
        applied, snap = random_neighbor(state, rng)
        if not applied:
            stayed += 1
            accept_hist.append(float('nan'))
        else:
            d_e = state.total_value - snap[1]
            p = math.exp(min(d_e / temperature, 700.0))
            accept_hist.append(p)
            if d_e >= 0 or rng.random() < p:
                if state.total_value > best_value:
                    best_value = state.total_value
                    best = state.copy()
                    best_iter = it + 1
            else:
                _restore(state, snap)
                stayed += 1
        it += 1
        temperature *= SA_ALPHA
        best_hist.append(best_value)
        cur_hist.append(state.total_value)
    return {'final_state': best, 'iterations': it, 'final_iter': best_iter,
            'stayed': stayed, 'best_hist': best_hist, 'cur_hist': cur_hist,
            'accept_hist': accept_hist}


# ============================================================================
# Algorithm 3: Genetic Algorithm (roulette + uniform crossover + repair)
# ============================================================================

def _crossover(problem, p1, p2, rng):
    """child is repaired into a valid state if necessary
    (greedy re-insertion by descending value)."""
    child = State(problem)
    n = len(problem.packets)
    sources = []
    wanted = []
    threshold = int(rng.random() * (n + 1))
    for i in range(n):
        src = p1 if i < threshold else p2
        child.orient[i] = src.orient[i]
        sources.append(src)
        wanted.append((src.inside[i], src.pos[i]))
    order = [i for i in range(n) if wanted[i][0]]
    rng.shuffle(order)
    order.sort(key=lambda i: problem.packets[i].value, reverse=True)
    for i in order:
        packet = problem.packets[i]
        if child.total_weight + packet.weight > problem.M:
            continue                                    # stays outside
        inside, pos = wanted[i]
        if try_place(child, i, True, pos, child.orient[i]) is not None:
            continue                                    # inherited placement works
        other = p2 if sources[i] is p1 else p1
        if other.inside[i] and try_place(child, i, True, other.pos[i],
                                         child.orient[i]) is not None:
            continue                                    # other parent's spot works
        for _ in range(GA_REPAIR_TRIES):
            cand = random_candidate(child, i, rng, child.orient[i])
            if cand is not None and try_place(child, i, True, cand,
                                              child.orient[i]) is not None:
                break                                   # greedy re-placement
    return child


def run_ga(problem, rng, pop_size, generations):
    """Generational GA with roulette selection, uniform crossover + repair,
    random-valid-move mutation and elitism."""
    population = [random_valid_state(problem, rng) for _ in range(pop_size)]
    initial_display = max(population, key=lambda s: s.total_value).copy()

    def fit(s):
        return s.total_value

    best = initial_display.copy()
    best_value = best.total_value
    best_gen = 0
    best_hist = [best_value]
    avg_hist = [sum(fit(s) for s in population) / pop_size]

    for gen in range(1, generations + 1):
        population.sort(key=fit, reverse=True)
        n_elite = min(GA_ELITE, pop_size)
        new_pop = [population[k].copy() for k in range(n_elite)]
        fits = [fit(s) for s in population]
        total = sum(fits)

        def roulette():
            if total <= 0:
                return population[rng.randrange(pop_size)]
            r = rng.uniform(0.0, total)
            acc = 0.0
            for s, f in zip(population, fits):
                acc += f
                if r <= acc:
                    return s
            return population[-1]

        while len(new_pop) < pop_size:
            parent1 = roulette()
            parent2 = roulette()
            child = _crossover(problem, parent1, parent2, rng)
            if rng.random() < GA_MUTATION_PROB:
                for _ in range(GA_MUTATION_TRIES):
                    applied, _snap = random_neighbor(child, rng)
                    if applied:
                        break                           # keep the first valid move
            new_pop.append(child)
        population = new_pop

        gen_best = max(fit(s) for s in population)
        if gen_best > best_value:
            best_value = gen_best
            best = max(population, key=fit).copy()
            best_gen = gen
        best_hist.append(best_value)
        avg_hist.append(sum(fit(s) for s in population) / pop_size)

    final_population = [(k + 1, s.total_value,
                         [problem.packets[i].pid for i in s._inside_indices()])
                        for k, s in enumerate(population)]
    return {'final_state': best, 'initial_display': initial_display,
            'iterations': generations, 'final_iter': best_gen,
            'best_hist': best_hist, 'avg_hist': avg_hist,
            'final_population': final_population,
            'pop_size': pop_size, 'generations': generations}

# ============================================================================
# Visualisation & reporting
# ============================================================================

_PLT = None      # matplotlib.pyplot module, set by init_matplotlib()


def init_matplotlib(interactive):
    """Select the matplotlib backend and import pyplot into _PLT."""
    global _PLT
    headless = sys.platform.startswith('linux') and not os.environ.get('DISPLAY')
    if not interactive or headless:
        matplotlib.use('Agg')
        if interactive and headless:
            print('(headless environment detected: interactive windows disabled, '
                  'output files only)')
    import matplotlib.pyplot as plt
    _PLT = plt


def save_curves(path, title, xlabel, ylabel, series, hline=None, logy=False):
    """Save one line chart. `series` = [(label, xs, ys, fmt), ...]."""
    fig, ax = _PLT.subplots(figsize=(9, 5), constrained_layout=True)
    for label, xs, ys, fmt in series:
        ax.plot(xs, ys, fmt, linewidth=1.3, label=label)
    if hline is not None:
        ax.axhline(hline, color='gray', linestyle='--', linewidth=1.0,
                   label='reference = %g' % hline)
    if logy:
        ax.set_yscale('log')
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    fig.savefig(path, dpi=150)
    _PLT.close(fig)


def _box_faces(b):
    """The 6 quad faces of a box (x0, y0, z0, x1, y1, z1)."""
    x0, y0, z0, x1, y1, z1 = b
    return [
        [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)],   # bottom
        [(x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1)],   # top
        [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0)],   # z = z0
        [(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)],   # z = z1
        [(x0, y0, z0), (x0, y1, z0), (x0, y1, z1), (x0, y0, z1)],   # x = x0
        [(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)],   # x = x1
    ]


def _v(x, y, z):
    """Visual remap for plotting: data (x, y=length-height-width axes) is drawn
    with the container HEIGHT on the vertical axis, i.e. plot coords are
    (x, z, y). This only affects the picture, never the stored state."""
    return (x, z, y)


def draw_state_3d(state, title):
    """Interactive 3D view of the container and all packets inside it.
    The figure is returned open (saved and shown by the caller)."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import matplotlib.patches as mpatches
    problem = state.problem
    L, H, W = problem.L, problem.H, problem.W

    fig = _PLT.figure(figsize=(9.5, 7.0), constrained_layout=True)
    ax = fig.add_subplot(111, projection='3d')

    edges = []
    for y in (0, H):
        edges += [((0, y, 0), (L, y, 0)), ((L, y, 0), (L, y, W)),
                  ((L, y, W), (0, y, W)), ((0, y, W), (0, y, 0))]
    for x in (0, L):
        for z in (0, W):
            edges.append(((x, 0, z), (x, H, z)))
    for a, b in edges:
        pa, pb = _v(*a), _v(*b)
        ax.plot([pa[0], pb[0]], [pa[1], pb[1]], [pa[2], pb[2]],
                color='black', linewidth=1.4)
    floor = Poly3DCollection([[_v(0, 0, 0), _v(L, 0, 0), _v(L, 0, W), _v(0, 0, W)]],
                             facecolors='lightgray', alpha=0.25, edgecolors='none')
    ax.add_collection3d(floor)

    inside = state._inside_indices()
    cmap = _PLT.get_cmap('tab20')
    for rank, i in enumerate(inside):
        b = state.box(i)
        packet = problem.packets[i]
        # remap the box: (x0, z0, y0, x1, z1, y1) in plot coordinates
        vb = (b[0], b[2], b[1], b[3], b[5], b[4])
        coll = Poly3DCollection(_box_faces(vb),
                                facecolors=cmap(rank % 20), alpha=0.55,
                                edgecolors='crimson' if packet.fragile else 'dimgray',
                                linewidths=1.0)
        ax.add_collection3d(coll)
        cx = (b[0] + b[3]) / 2.0
        cy = (b[1] + b[4]) / 2.0
        cz = (b[2] + b[5]) / 2.0
        ax.text(*_v(cx, cy, cz), packet.pid, size=7, ha='center', va='center',
                color='black', fontweight='bold')

    ax.set_xlim(0, L)
    ax.set_ylim(0, W)
    ax.set_zlim(0, H)
    try:
        ax.set_box_aspect((L, W, H))
    except AttributeError:               # older matplotlib
        pass
    ax.set_xlabel('x (length)')
    ax.set_ylabel('z (width)')
    ax.set_zlabel('y (height)')
    ax.set_title('%s\ntotal value inside: %d    packets inside: %d/%d'
                 % (title, state.total_value, len(inside), len(problem.packets)))
    handles = [mpatches.Patch(facecolor='steelblue', alpha=0.55,
                              edgecolor='dimgray', label='packet'),
               mpatches.Patch(facecolor='crimson', alpha=0.55,
                              edgecolor='crimson', label='fragile packet')]
    ax.legend(handles=handles, loc='upper right')
    ax.view_init(elev=22, azim=-55)
    return fig


def state_listing(state, header):
    """Text listing of a state: every packet with its inside/outside status."""
    problem = state.problem
    lines = [header]
    inside = state._inside_indices()
    lines.append('Packets INSIDE (%d) - total value %d, total weight %d/%d:'
                 % (len(inside), state.total_value, state.total_weight, problem.M))
    for i in inside:
        p = problem.packets[i]
        o = state.orient[i]
        lines.append('  [%-10s] pos=%s  size=%s  orient=(l->%s, h->%s, w->%s)  '
                     'value=%d  weight=%d  fragile=%s'
                     % (p.pid, state.pos[i], state.dims(i), o[0], o[1], o[2],
                        p.value, p.weight, 'yes' if p.fragile else 'no'))
    outside = [i for i in range(len(problem.packets)) if not state.inside[i]]
    lines.append('Packets OUTSIDE (%d) - parked at %s:'
                 % (len(outside), problem.parking))
    for i in outside:
        p = problem.packets[i]
        lines.append('  [%-10s] value=%d  weight=%d  fragile=%s'
                     % (p.pid, p.value, p.weight, 'yes' if p.fragile else 'no'))
    return '\n'.join(lines)


INPUT_FORMAT_HELP = """\
input file format:
    L H W M                    container dimensions + maximum weight
    T [Op1 Op2]                1=Steepest Ascent HC, 2=Simulated Annealing,
                               3=Genetic Algorithm (Op1=population, Op2=generations)
    N                          number of packets
    id l h w ol oh ow v we f   one line per packet; ol oh ow = axis each of
                               length/height/width is parallel to (x/y/z);
                               f = fragile flag (0/1)
Blank lines and '#' comments are ignored."""


# ============================================================================
# Main
# ============================================================================

def main(argv=None):
    parser = argparse.ArgumentParser(
        prog='packet_packer',
        description='3D container packet packing via Steepest Ascent Hill '
                    'Climbing, Simulated Annealing (geometric cooling) or a '
                    'Genetic Algorithm.',
        epilog=INPUT_FORMAT_HELP,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('input', help='path to the input text file')
    parser.add_argument('--outdir', default='output',
                        help='directory for output files (default: ./output)')
    parser.add_argument('--seed', type=int, default=None,
                        help='random seed for reproducible runs')
    parser.add_argument('--no-show', action='store_true',
                        help='do not open interactive windows')
    args = parser.parse_args(argv)

    try:
        problem, algo_type, op1, op2 = parse_input(args.input)
    except InputError as exc:
        parser.exit(2, 'error: %s\n' % exc)

    init_matplotlib(interactive=not args.no_show)
    rng = random.Random(args.seed)

    t_start = time.perf_counter()
    if algo_type == 1:
        initial = random_valid_state(problem, rng)
        result = run_hc(initial.copy(), rng)
        label = 'Steepest Ascent Hill Climbing'
    elif algo_type == 2:
        initial = random_valid_state(problem, rng)
        result = run_sa(initial.copy(), rng)
        label = 'Simulated Annealing (geometric cooling)'
    else:
        pop_size = op1 if op1 is not None else GA_DEFAULT_POP
        generations = op2 if op2 is not None else GA_DEFAULT_GEN
        if pop_size <= 0 or generations <= 0:
            print('error: GA population and generation counts must be positive',
                  file=sys.stderr)
            return 2
        result = run_ga(problem, rng, pop_size, generations)
        initial = result['initial_display']
        label = 'Genetic Algorithm (roulette + uniform crossover)'
    elapsed = time.perf_counter() - t_start
    final = result['final_state']

    for name, st in (('initial', initial), ('final', final)):
        if not st.validate_full():
            print('error: %s state failed the validity check (internal error)'
                  % name, file=sys.stderr)
            return 3

    # ---------------- console report ----------------
    print('=' * 78)
    print('3D CONTAINER PACKING - %s' % label)
    print('=' * 78)
    print('Container: L=%d H=%d W=%d, max weight=%d    packets: %d'
          % (problem.L, problem.H, problem.W, problem.M, len(problem.packets)))
    if args.seed is not None:
        print('Random seed: %d' % args.seed)
    print()

    if algo_type == 3:
        initial_listing = state_listing(initial, '--- INITIAL STATE '
                                        '(best individual of the starting population) ---')
    else:
        initial_listing = state_listing(initial, '--- INITIAL STATE ---')
    final_listing = state_listing(final, '--- FINAL STATE ---')
    print(initial_listing)
    print()
    print('Total value of the initial state: %d' % initial.total_value)
    print()

    print('--- ALGORITHM RUN ---')
    if algo_type == 2:
        print('Parameters: T0=%g, alpha=%g, max iterations=%d'
              % (SA_T0, SA_ALPHA, SA_MAX_ITER))
    elif algo_type == 3:
        note = '' if (op1 is not None and op2 is not None) else \
               ' (defaults applied for missing Op1/Op2)'
        print('Parameters: starting population=%d, generations=%d%s'
              % (result['pop_size'], result['generations'], note))
    else:
        print('Parameters: strict improvement only, stop at first local optimum')
    print('Processing time: %.3f s' % elapsed)
    print('Iterations executed: %d' % result['iterations'])
    print('Final state first reached at iteration: %d' % result['final_iter'])
    if algo_type == 2:
        print('Times stayed in the current state: %d' % result['stayed'])
    print()

    if algo_type == 3:
        pop_lines = ['--- FINAL POPULATION (%d individuals) ---'
                     % result['pop_size']]
        best_row = max(result['final_population'], key=lambda t: t[1])
        for idx, value, ids in result['final_population']:
            mark = '  <- best' if idx == best_row[0] else ''
            pop_lines.append('  #%-4d value=%-6d inside: %s%s'
                             % (idx, value, ' '.join(ids) if ids else '-', mark))
        for line in pop_lines:
            print(line)
        print()

    print(final_listing)
    print()
    improvement = final.total_value - initial.total_value
    if initial.total_value > 0:
        pct = 100.0 * improvement / initial.total_value
        print('Total value of the final state: %d '
              '(initial %d, improvement +%d, +%.1f%%)'
              % (final.total_value, initial.total_value, improvement, pct))
    else:
        print('Total value of the final state: %d (initial %d, improvement +%d)'
              % (final.total_value, initial.total_value, improvement))

    # ---------------- output files ----------------
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)
    saved = []

    def save_text(name, text):
        path = os.path.join(outdir, name)
        with open(path, 'w', encoding='utf-8') as fh:
            fh.write(text + '\n')
        saved.append(path)

    save_text('initial_state.txt', initial_listing)
    save_text('final_state.txt', final_listing)

    fig = draw_state_3d(initial, 'Initial state')
    path = os.path.join(outdir, 'container_initial.png')
    fig.savefig(path, dpi=150)
    saved.append(path)
    fig = draw_state_3d(final, 'Final state')
    path = os.path.join(outdir, 'container_final.png')
    fig.savefig(path, dpi=150)
    saved.append(path)

    if algo_type == 1:
        hist = result['history']
        save_curves(os.path.join(outdir, 'value_curve.png'),
                    'Steepest Ascent Hill Climbing - total value inside vs iteration',
                    'iteration', 'total value inside',
                    [('total value inside', list(range(len(hist))), hist, 'b-')])
        saved.append(os.path.join(outdir, 'value_curve.png'))
    elif algo_type == 2:
        best_hist = result['best_hist']
        cur_hist = result['cur_hist']
        iters = list(range(len(best_hist)))
        save_curves(os.path.join(outdir, 'value_curve.png'),
                    'Simulated Annealing - value inside vs iteration',
                    'iteration', 'total value inside',
                    [('best value so far', iters, best_hist, 'b-'),
                     ('current state value', iters, cur_hist, 'r-')])
        saved.append(os.path.join(outdir, 'value_curve.png'))
        acc = result['accept_hist']
        save_curves(os.path.join(outdir, 'sa_acceptance.png'),
                    'Simulated Annealing - acceptance factor e^(dE/T) vs iteration',
                    'iteration', 'e^(dE/T)',
                    [('e^(dE/T)', list(range(1, len(acc) + 1)), acc, 'g-')],
                    hline=1.0, logy=True)
        saved.append(os.path.join(outdir, 'sa_acceptance.png'))
    else:
        best_hist = result['best_hist']
        avg_hist = result['avg_hist']
        gens = list(range(len(best_hist)))
        save_curves(os.path.join(outdir, 'ga_best_curve.png'),
                    'Genetic Algorithm - best value in population vs generation',
                    'generation', 'best total value inside',
                    [('best current candidate', gens, best_hist, 'b-')])
        save_curves(os.path.join(outdir, 'ga_avg_curve.png'),
                    'Genetic Algorithm - average population value vs generation',
                    'generation', 'average total value inside',
                    [('population average', gens, avg_hist, 'g-')])
        saved.append(os.path.join(outdir, 'ga_best_curve.png'))
        saved.append(os.path.join(outdir, 'ga_avg_curve.png'))
        if algo_type == 3:
            save_text('final_population.txt', '\n'.join(pop_lines))

    print()
    print('Output files:')
    for p in saved:
        print('  %s' % p)

    if not args.no_show and matplotlib.get_backend().lower() != 'agg':
        print()
        print('Opening interactive windows (close them to finish)...')
        try:
            _PLT.show()
        except Exception as exc:                      # pragma: no cover
            print('(could not open interactive windows: %s)' % exc)
    return 0


if __name__ == '__main__':
    sys.exit(main())
