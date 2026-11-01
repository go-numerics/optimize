// This file is a Go port of DIRect.c from SciPy v1.18.1
// (scipy/optimize/_direct/), itself the NLopt/Johnson cleanup of J. M.
// Gablonsky's DIRECT 2.0.4 Fortran code.
//
// Copyright (c) 1999, 2000, 2001 North Carolina State University
//
// This program is distributed under the MIT License (see COPYING.DIRECT in
// the root of this repository, which carries the full text):
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of the Software, to deal in the Software without restriction, subject
// to the above copyright notice and this permission notice being included in
// all copies or substantial portions of the Software. THE SOFTWARE IS
// PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.

package optimize

import (
	"math"
	"sync"
)

// dirMaxMemory is MAXMEMORY of DIRect.c.
const dirMaxMemory = 1 << 30

// dirParams are the arguments the Python layer hands to direct_direct_ after
// direct_optimize has normalised them (volper, sigmaper <= 0 become -1).
type dirParams struct {
	eps      float64
	maxf     int64 // user's maxfun; <= 0 selects the C default of 100000
	maxt     int64 // user's maxiter; <= 0 selects the C default
	original bool
	fglobal  float64
	fglper   float64
	volper   float64
	sigmaper float64
	workers  int
	// deepCap lowers MAXDEEP; used by tests to reach the "maximum number of
	// levels" exit, which MaxIter+1000 puts out of reach of cheap runs.
	deepCap int
}

// dirOut is what _directmodule.c returns: x, minf, ret_code, numfunc,
// numiter.
type dirOut struct {
	x       []float64
	fun     float64
	ierror  int
	numfunc int
	numiter int
}

// dirSizes reproduces the sizing arithmetic at the top of direct_direct_
// (DIRect.c:94-145) in int64: MAXFUNC and MAXDEEP, the memory cap, and the
// adjusted budgets maxf and maxT.
func dirSizes(n int, maxf, maxt int64) (maxfunc, maxfAdj, maxdeep, maxtAdj int64) {
	if maxf <= 0 {
		maxfunc = 101000
	} else {
		maxfunc = maxf + 1000 + maxf/2
	}
	dim := int64(n)*(8+4) + (8*2 + 4)
	if maxfunc*dim > dirMaxMemory {
		maxfunc = dirMaxMemory / dim
	}
	if maxf <= 0 {
		maxfAdj = maxfunc - 1000
	} else {
		maxfAdj = 2 * (maxfunc - 1000) / 3
	}
	if maxt <= 0 {
		maxdeep = maxfunc / 5
	} else {
		maxdeep = maxt + 1000
	}
	const dimDeep, constMem = 8*2 + 4, 2 * (8 + 4)
	if maxdeep*dimDeep+constMem > dirMaxMemory {
		maxdeep = (dirMaxMemory - constMem) / dimDeep
	}
	if maxt <= 0 {
		maxtAdj = maxdeep
	} else {
		maxtAdj = maxdeep - 1000
	}
	return
}

// directCore is direct_direct_. fn returns the value and the feasibility
// flag of DIRinfcn (0 feasible, 1 infeasible); SciPy always reports 0. lb and
// ub must satisfy lb < ub. The callback gets the last evaluated point, the
// best point and its value, and returns true to stop (status -102).
func directCore(fn func(x []float64) (float64, int), lb, ub []float64, p dirParams,
	callback func(xLast, xBest []float64, fBest float64) bool) dirOut {

	n := len(lb)
	maxfunc, maxf64, maxdeep, maxT64 := dirSizes(n, p.maxf, p.maxt)
	if p.deepCap > 0 && int64(p.deepCap) < maxdeep {
		maxdeep = int64(p.deepCap)
	}
	maxf, maxT := int(maxf64), int(maxT64)
	d := &dirState{
		n: n, maxfunc: int(maxfunc), maxdeep: int(maxdeep), fn: fn, workers: p.workers,
		fv: make([]float64, 1), fk: make([]int8, 1), point: make([]int32, 1),
		anchor: make([]int32, 8),
		s1:     make([]int, dirMaxDiv+2), s2: make([]int, dirMaxDiv+2),
		arrayi: make([]int, n+1), w: make([]float64, n+1),
		l2next: make([]int, n+1), l2pos: make([]int, n+1),
		xs1: make([]float64, n), xs2: make([]float64, n),
		xLast: make([]float64, n), xbuf: make([]float64, n),
	}
	d.jones = 1
	if p.original {
		d.jones = 0
	}
	const epsabs = 0.0
	eps := p.eps
	epsfix := 1e100
	iepschange := false
	if eps < 0 {
		iepschange = true
		epsfix = -eps
		eps = -eps
	}
	// DIRpreprc: the box is validated by the caller (lb < ub, finite range).
	for i := 0; i < n; i++ {
		help := ub[i] - lb[i]
		d.xs2[i] = lb[i] / help
		d.xs1[i] = help
	}
	// bestX is the returned x of DIRect.c:805, a different rounding from the
	// points handed to the objective.
	bestX := func() []float64 {
		x := make([]float64, n)
		off := d.cOff(d.minpos)
		for i := range x {
			x[i] = float64(d.c[off+i]*d.xs1[i]) + float64(d.xs1[i]*d.xs2[i])
		}
		return x
	}
	// cleanup: x is the last evaluated point, as SciPy's x_seq.
	fail := func(ierror, t int) dirOut {
		x := make([]float64, n)
		copy(x, d.xLast)
		return dirOut{x: x, fun: d.minf, ierror: ierror, numfunc: d.numfunc, numiter: t}
	}
	l100 := func(ierror, t int) dirOut {
		return dirOut{x: bestX(), fun: d.minf, ierror: ierror, numfunc: d.numfunc, numiter: t}
	}

	// DIRheader's `maxf + 20 > MAXFUNC` error (-2) cannot happen after the
	// sizing above: maxf is then 2*(MAXFUNC-1000)/3 or MAXFUNC-1000, which
	// is below MAXFUNC-20 for every MAXFUNC that dirSizes can return, so the
	// check is not ported.
	divfactor := 1.0
	if p.fglobal != 0 {
		divfactor = math.Abs(p.fglobal)
	}
	oldmaxf := maxf
	increase := false

	// DIRinitlist / DIRinit
	d.initTables(maxdeep)
	d.growRects(1)
	for i := 0; i < n; i++ {
		d.c[i] = 0.5
	}
	d.scalePoint(1, d.xbuf)
	f1, kret := fn(d.xbuf)
	copy(d.xLast, d.xbuf)
	d.fv[1] = f1
	d.fk[1] = int8(kret)
	d.iinfesible = kret
	d.fmax = f1
	if d.fk[1] > 0 {
		d.fv[1] = math.Inf(1)
		d.fmax = d.fv[1]
		d.ifeasiblef = 1
	}
	d.minf = d.fv[1]
	d.minpos = 1
	d.point[1] = 0
	d.free = 2
	delta := d.thirds[1]
	d.getI(1)
	// DIRSamplepoints fails when the 2n new slots would end the free list;
	// checked here so that the arrays are never grown for a doomed start.
	if 2*n+2 > d.maxfunc {
		d.numfunc = 1
		return fail(-4, 0)
	}
	start, _ := d.samplepoints(delta, 1)
	d.samplef(start)
	d.divide(start, 0, 1)
	d.insertlist(start, 1)

	d.numfunc = d.maxi + 1 + d.maxi
	actmaxdeep := 1
	t := 2
	mdeep := d.maxdeep
	for t = 2; t <= maxT-1; t++ {
		maxpos, ok := d.choose(eps, epsabs)
		if !ok {
			return l100(-6, t)
		}
		if d.jones == 0 && !d.doubleInsert(&maxpos) {
			return fail(-6, t)
		}
		for j := 1; j <= maxpos; j++ {
			if d.s1[j] > 0 {
				actdeepDiv := d.getmaxdeep(d.s1[j])
				delta = tableAt(d.thirds, actdeepDiv+1)
				actdeep := d.s2[j]
				if actdeep+1 >= mdeep {
					return l100(-6, t)
				}
				if actdeep > actmaxdeep {
					actmaxdeep = actdeep
				}
				help := d.s1[j]
				if d.anc(actdeep) != help {
					pos1 := d.anc(actdeep)
					for int(d.point[pos1]) != help {
						pos1 = int(d.point[pos1])
					}
					d.point[pos1] = d.point[help]
				} else {
					d.setAnc(actdeep, int(d.point[help]))
				}
				d.getI(help)
				start, ok = d.samplepoints(delta, help)
				if !ok {
					return fail(-4, t)
				}
				d.samplef(start)
				d.divide(start, actdeepDiv, help)
				d.insertlist(start, help)
				d.numfunc += d.maxi + d.maxi
			}
		}
		// Termination checks.
		delta = tableAt(d.thirds, d.getlevel(d.minpos, 0))
		if delta <= p.volper {
			return l100(4, t)
		}
		delta = d.lev(d.getlevel(d.minpos, d.jones))
		if delta <= p.sigmaper {
			return l100(5, t)
		}
		if (d.minf-p.fglobal)/divfactor <= p.fglper {
			return l100(3, t)
		}
		if d.iinfesible > 0 {
			d.replaceinf()
		}
		if iepschange {
			eps = dirMax(math.Abs(d.minf)*1e-4, epsfix)
		}
		if increase {
			maxf = d.numfunc + oldmaxf
			if d.ifeasiblef == 0 {
				increase = false
			}
		}
		if d.numfunc > maxf {
			if d.ifeasiblef == 0 {
				return l100(1, t)
			}
			increase = true
			maxf = d.numfunc + oldmaxf
		}
		if callback != nil {
			xl := make([]float64, n)
			copy(xl, d.xLast)
			if callback(xl, bestX(), d.minf) {
				return l100(-102, t)
			}
		}
	}
	return l100(2, t)
}

// initTables fills thirds and levels as DIRinit does. 1/3^i is exactly 0
// from i = 647 on (3^647 overflows to +Inf), so the tables stop at 700.
func (d *dirState) initTables(maxdeep int64) {
	nt := int(maxdeep)
	if nt > 700 {
		nt = 700
	}
	d.thirds = make([]float64, nt+1)
	d.thirds[0] = 1
	help2 := 3.0
	for i := 1; i <= nt; i++ {
		d.thirds[i] = 1 / help2
		help2 *= 3
	}
	if d.jones == 1 {
		return
	}
	n := d.n
	d.w0 = make([]float64, n)
	for j := 0; j < n; j++ {
		d.w0[j] = math.Sqrt(float64(n-j)+float64(j)/9.) * .5
	}
	d.levelBlocks = int(maxdeep) / n
	nb := d.levelBlocks
	if nb > 700 {
		nb = 700
	}
	d.pow3 = make([]float64, nb)
	help2 = 1
	for i := 0; i < nb; i++ {
		d.pow3[i] = help2
		help2 *= 3
	}
}

// lev is levels[i] of the C code: 1/3^i for DIRECT-L, and the half
// diagonal w_j / 3^k of a rectangle of level k*n+j for the original DIRECT.
func (d *dirState) lev(i int) float64 {
	if d.jones == 1 {
		return tableAt(d.thirds, i)
	}
	b := i / d.n
	if b >= d.levelBlocks || b >= len(d.pow3) {
		return 0
	}
	return d.w0[i%d.n] / d.pow3[b]
}

// evalParallel evaluates the objective at the points in d.posBuf on
// d.workers goroutines. A panic in the objective is re-raised on the calling
// goroutine.
func (d *dirState) evalParallel(vals []float64, flags []int) {
	cnt := len(vals)
	w := d.workers
	if w > cnt {
		w = cnt
	}
	var wg sync.WaitGroup
	var mu sync.Mutex
	var pnc any
	panicked := false
	for g := 0; g < w; g++ {
		lo, hi := g*cnt/w, (g+1)*cnt/w
		wg.Add(1)
		go func() {
			defer wg.Done()
			defer func() {
				if r := recover(); r != nil {
					mu.Lock()
					if !panicked {
						panicked, pnc = true, r
					}
					mu.Unlock()
				}
			}()
			x := make([]float64, d.n)
			for j := lo; j < hi; j++ {
				d.scalePoint(d.posBuf[j], x)
				vals[j], flags[j] = d.fn(x)
			}
		}()
	}
	wg.Wait()
	if panicked {
		panic(pnc)
	}
}
