// This file is a Go port of DIRsubrout.c and DIRserial.c from SciPy v1.18.1
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

import "math"

// dirMaxDiv is MAXDIV of DIRect.c: the capacity of the list S of
// potentially optimal rectangles.
const dirMaxDiv = 5000

// dirState holds the arrays of DIRect.c. The C code indexes rectangles,
// dimensions and list slots from 1; the slices below keep that numbering
// (element 0 of a 1-based slice is unused) and the accessors translate the
// (position, dimension) pairs of the Fortran arrays c(n, maxfunc) and
// length(n, maxfunc).
//
// The rectangle arrays are allocated lazily up to maxfunc slots, which is
// the size the C code mallocs at once: the free list never recycles a slot,
// so the slots are handed out in order and the arrays only need to cover the
// slots handed out so far. The lists keyed by depth (anchor, thirds, levels)
// are likewise sized by use; thirds and levels are exactly 0 from index ~650
// on (1/3^i underflows), so a read past the table is 0.
type dirState struct {
	n     int
	jones int // 1: DIRECT-L (Gablonsky), 0: Jones' original

	maxfunc int // MAXFUNC
	maxdeep int // MAXDEEP

	c      []float64 // c[(pos-1)*n + dim-1]
	length []int32   // length[(pos-1)*n + dim-1]
	fv     []float64 // f(1, pos), 1-based
	fk     []int8    // f(2, pos): 0 feasible, 1 replaced, 2 infeasible, 1-based
	point  []int32   // the lists, 1-based; point[maxfunc] = 0
	free   int       // first free position

	anchor []int32 // anchor of the list of depth d is anchor[d+1]; anchor[0] is the NaN list
	thirds []float64
	// levels of the original DIRECT: w0[j] / pow3[k] for level k*n+j (lev).
	w0          []float64
	pow3        []float64
	levelBlocks int

	s1, s2 []int // the list S: positions and levels, 1-based

	arrayi []int // dimensions of maximal side length, 1-based
	maxi   int
	w      []float64 // 1-based
	l2next []int     // list2(j, 1), 1-based
	l2pos  []int     // list2(j, 2), 1-based

	xs1, xs2 []float64 // scaling: l and u of the C code after DIRpreprc
	fn       func(x []float64) (float64, int)
	workers  int

	minf       float64
	minpos     int
	fmax       float64
	ifeasiblef int
	iinfesible int
	numfunc    int

	xLast   []float64 // x_seq: the last point handed to the objective
	xbuf    []float64
	posBuf  []int
	valBuf  []float64
	flagBuf []int
}

func (d *dirState) cOff(pos int) int { return (pos - 1) * d.n }

// growRects makes slots 1..upto addressable (upto is clamped to maxfunc).
func (d *dirState) growRects(upto int) {
	if upto > d.maxfunc {
		upto = d.maxfunc
	}
	have := len(d.point) - 1
	if upto <= have {
		return
	}
	want := 2 * have
	if want < upto {
		want = upto
	}
	if want > d.maxfunc {
		want = d.maxfunc
	}
	n := d.n
	d.c = append(d.c, make([]float64, (want-have)*n)...)
	d.length = append(d.length, make([]int32, (want-have)*n)...)
	d.fv = append(d.fv, make([]float64, want-have)...)
	d.fk = append(d.fk, make([]int8, want-have)...)
	for i := have + 1; i <= want; i++ {
		d.point = append(d.point, int32(i+1))
	}
	if want == d.maxfunc {
		d.point[want] = 0
	}
}

// anc returns anchor[level] of the C code (level >= 0).
func (d *dirState) anc(level int) int {
	if level+1 < len(d.anchor) {
		return int(d.anchor[level+1])
	}
	return 0
}

func (d *dirState) setAnc(level, v int) {
	if level+1 >= len(d.anchor) {
		want := 2 * len(d.anchor)
		if want < level+2 {
			want = level + 2
		}
		d.anchor = append(d.anchor, make([]int32, want-len(d.anchor))...)
	}
	d.anchor[level+1] = int32(v)
}

func tableAt(t []float64, i int) float64 {
	if i < len(t) {
		return t[i]
	}
	return 0
}

// cmin and cmax are the C macros MIN and MAX: the first argument wins ties
// and NaNs, which math.Min and math.Max do not do.
func dirMin(a, b float64) float64 {
	if a < b {
		return a
	}
	return b
}

func dirMax(a, b float64) float64 {
	if a > b {
		return a
	}
	return b
}

// getlevel is DIRGetlevel: the level of the rectangle at pos.
func (d *dirState) getlevel(pos, jones int) int {
	n := d.n
	l := d.length[d.cOff(pos) : d.cOff(pos)+n]
	if jones == 0 {
		help := int(l[0])
		k := help
		p := 1
		for i := 1; i < n; i++ {
			if int(l[i]) < k {
				k = int(l[i])
			}
			if int(l[i]) == help {
				p++
			}
		}
		if k == help {
			return k*n + n - p
		}
		return k*n + p
	}
	return d.getmaxdeep(pos)
}

// getmaxdeep is DIRGetmaxDeep: the smallest side index of the rectangle.
func (d *dirState) getmaxdeep(pos int) int {
	l := d.length[d.cOff(pos) : d.cOff(pos)+d.n]
	help := l[0]
	for i := 1; i < len(l); i++ {
		if l[i] < help {
			help = l[i]
		}
	}
	return int(help)
}

// choose is DIRChoose: it fills S with the potentially optimal rectangles
// and returns their number. ok is false when S would overflow (the C code
// writes out of bounds there).
//
// The C code ends the "helpgreater > helplower" branch with
// `if (logfile) goto L60;`, and SciPy passes no logfile, so that branch
// removes nothing: it is kept as the empty else it is in practice.
// The NaN list anchor[-1] (rectangles of undefined value) is never filled by
// DIRInsertList, so the C code's `novalue` entry is always absent and is not
// ported.
func (d *dirState) choose(epsrel, epsabs float64) (maxpos int, ok bool) {
	k := 1
	if d.ifeasiblef >= 1 {
		for j := 0; j < len(d.anchor)-1; j++ {
			if a := d.anc(j); a > 0 {
				d.s1[1] = a
				d.s2[1] = d.getlevel(a, d.jones)
				break
			}
		}
		return 1, true
	}
	for j := 0; j < len(d.anchor)-1; j++ {
		if a := d.anc(j); a > 0 {
			if k > dirMaxDiv {
				return 0, false
			}
			d.s1[k] = a
			d.s2[k] = d.getlevel(a, d.jones)
			k++
		}
	}
	maxpos = k - 1
	if k-1 <= d.maxdeep {
		d.s1[k] = 0
	}
	minf := d.minf
	for j := maxpos; j >= 1; j-- {
		helplower := math.Inf(1)
		helpgreater := 0.0
		jj := d.s1[j]
		drop := false
		for i := 1; i <= j-1 && !drop; i++ {
			ii := d.s1[i]
			if ii > 0 && d.fk[ii] <= 1 {
				help2 := d.lev(d.s2[i]) - d.lev(d.s2[j])
				help2 = (d.fv[ii] - d.fv[jj]) / help2
				if help2 <= 0 {
					drop = true
				} else if help2 < helplower {
					helplower = help2
				}
			}
		}
		for i := j + 1; i <= maxpos && !drop; i++ {
			ii := d.s1[i]
			if ii > 0 && d.fk[ii] <= 1 {
				help2 := d.lev(d.s2[i]) - d.lev(d.s2[j])
				help2 = (d.fv[ii] - d.fv[jj]) / help2
				if help2 <= 0 {
					drop = true
				} else if help2 > helpgreater {
					helpgreater = help2
				}
			}
		}
		if !drop && helpgreater <= helplower {
			lhs := d.fv[jj] - float64(helplower*d.lev(d.s2[j]))
			if lhs > dirMin(minf-float64(epsrel*math.Abs(minf)), minf-epsabs) {
				drop = true
			}
		}
		if drop {
			d.s1[j] = 0
		}
	}
	return maxpos, true
}

// doubleInsert is DIRDoubleInsert (original DIRECT only): it adds to S every
// rectangle of the same level whose centre value equals that of a selected
// one. It reports false when S is full (status -6).
func (d *dirState) doubleInsert(maxpos *int) bool {
	oldmaxpos := *maxpos
	for i := 1; i <= oldmaxpos; i++ {
		if d.s1[i] > 0 {
			actdeep := d.s2[i]
			help := d.anc(actdeep)
			pos := int(d.point[help])
			iflag := 0
			for pos > 0 && iflag == 0 {
				if d.fv[pos]-d.fv[help] <= 1e-13 {
					if *maxpos < dirMaxDiv {
						*maxpos++
						d.s1[*maxpos] = pos
						d.s2[*maxpos] = actdeep
						pos = int(d.point[pos])
					} else {
						return false
					}
				} else {
					iflag = 1
				}
			}
		}
	}
	return true
}

// isinbox is isinbox_: whether x lies in the box [a, b].
func isinbox(x, a, b []float64) bool {
	for i := range x {
		if a[i] > x[i] || b[i] < x[i] {
			return false
		}
	}
	return true
}

// resortlist is dirresortlist_: it re-inserts the rectangle `replace`, whose
// value was replaced, into its list.
func (d *dirState) resortlist(replace int) {
	l := d.getlevel(replace, d.jones)
	start := d.anc(l)
	if replace == start {
		return
	}
	pos := start
	for i := 1; i <= d.maxfunc; i++ {
		if int(d.point[pos]) == replace {
			d.point[pos] = d.point[replace]
			break
		}
		pos = int(d.point[pos])
		if pos == 0 {
			break
		}
	}
	if d.fv[start] > d.fv[replace] {
		d.setAnc(l, replace)
		d.point[replace] = int32(start)
		return
	}
	pos = start
	for i := 1; i <= d.maxfunc; i++ {
		if d.point[pos] == 0 {
			d.point[replace] = d.point[pos]
			d.point[pos] = int32(replace)
			return
		}
		if d.fv[d.point[pos]] > d.fv[replace] {
			d.point[replace] = d.point[pos]
			d.point[pos] = int32(replace)
			return
		}
		pos = int(d.point[pos])
	}
}

// replaceinf is DIRreplaceInf: infeasible points near feasible ones take
// the lowest nearby value. SciPy never marks a point infeasible, so the
// public API cannot reach it; it is reachable through the internal flag and
// is tested there.
func (d *dirState) replaceinf() {
	n := d.n
	a := make([]float64, n)
	b := make([]float64, n)
	for i := 1; i <= d.free-1; i++ {
		if d.fk[i] > 0 {
			for j := 0; j < n; j++ {
				sidelength := tableAt(d.thirds, int(d.length[d.cOff(i)+j]))
				a[j] = d.c[d.cOff(i)+j] - sidelength
				b[j] = d.c[d.cOff(i)+j] + sidelength
			}
			d.fv[i] = math.Inf(1)
			d.fk[i] = 2
			for k := 1; k <= d.free-1; k++ {
				if d.fk[k] == 0 {
					if isinbox(d.c[d.cOff(k):d.cOff(k)+n], a, b) {
						d.fv[i] = dirMin(d.fv[i], d.fv[k])
						d.fk[i] = 1
					}
				}
			}
			if d.fk[i] == 1 {
				// 1e-6f is a C float constant.
				d.fv[i] += float64(math.Abs(d.fv[i]) * float64(float32(1e-6)))
				d.resortlist(i)
			} else if !(d.fmax == d.fv[i]) {
				d.fv[i] = dirMax(d.fmax+1, d.fv[i])
			}
		}
	}
}

// insert is DIRInsert: it inserts ins into the sorted list after start.
// start is advanced as the C code advances its by-reference argument.
func (d *dirState) insert(start *int, ins int) {
	for i := 1; i <= d.maxfunc; i++ {
		if d.point[*start] == 0 {
			d.point[*start] = int32(ins)
			d.point[ins] = 0
			return
		} else if d.fv[ins] < d.fv[d.point[*start]] {
			help := d.point[*start]
			d.point[*start] = int32(ins)
			d.point[ins] = help
			return
		}
		*start = int(d.point[*start])
	}
}

// insertlist is DIRInsertList: it inserts the maxi pairs of new rectangles
// starting at new, and then the divided rectangle samp, into the lists.
func (d *dirState) insertlist(new_, samp int) {
	for j := 1; j <= d.maxi; j++ {
		pos1 := new_
		pos2 := int(d.point[pos1])
		new_ = int(d.point[pos2])
		deep := d.getlevel(pos1, d.jones)
		if d.anc(deep) == 0 {
			if d.fv[pos2] < d.fv[pos1] {
				d.setAnc(deep, pos2)
				d.point[pos2] = int32(pos1)
				d.point[pos1] = 0
			} else {
				d.setAnc(deep, pos1)
				d.point[pos2] = 0
			}
		} else {
			pos := d.anc(deep)
			if d.fv[pos2] < d.fv[pos1] {
				if d.fv[pos2] < d.fv[pos] {
					d.setAnc(deep, pos2)
					if d.fv[pos1] < d.fv[pos] {
						d.point[pos2] = int32(pos1)
						d.point[pos1] = int32(pos)
					} else {
						d.point[pos2] = int32(pos)
						d.insert(&pos, pos1)
					}
				} else {
					d.insert(&pos, pos2)
					d.insert(&pos, pos1)
				}
			} else {
				if d.fv[pos1] < d.fv[pos] {
					d.setAnc(deep, pos1)
					if d.fv[pos] < d.fv[pos2] {
						d.point[pos1] = int32(pos)
						d.insert(&pos, pos2)
					} else {
						d.point[pos1] = int32(pos2)
						d.point[pos2] = int32(pos)
					}
				} else {
					d.insert(&pos, pos1)
					d.insert(&pos, pos2)
				}
			}
		}
	}
	deep := d.getlevel(samp, d.jones)
	pos := d.anc(deep)
	if d.fv[samp] < d.fv[pos] {
		d.setAnc(deep, samp)
		d.point[samp] = int32(pos)
	} else {
		d.insert(&pos, samp)
	}
}

// insertlist2 is dirinsertlist_2__: it inserts dimension j (whose smaller
// sample value is w[j] and whose first sample is at k) into the list of
// dimensions sorted by w.
func (d *dirState) insertlist2(start *int, j, k int) {
	pos := *start
	if *start == 0 {
		d.l2next[j] = 0
		*start = j
	} else if d.w[*start] > d.w[j] {
		d.l2next[j] = *start
		*start = j
	} else {
		for i := 1; i <= d.maxi; i++ {
			if d.l2next[pos] == 0 {
				d.l2next[j] = 0
				d.l2next[pos] = j
				break
			} else if d.w[j] < d.w[d.l2next[pos]] {
				d.l2next[j] = d.l2next[pos]
				d.l2next[pos] = j
				break
			}
			pos = d.l2next[pos]
		}
	}
	d.l2pos[j] = k
}

// samplepoints is DIRSamplepoints: it takes 2*maxi free slots, copies the
// rectangle `sample` into them and offsets the centres by +-delta along the
// maxi longest sides. It reports false when the free list runs out.
func (d *dirState) samplepoints(delta float64, sample int) (start int, ok bool) {
	n := d.n
	d.growRects(d.free + 2*d.maxi)
	start = d.free
	pos := d.free
	for k := 1; k <= 2*d.maxi; k++ {
		copy(d.length[d.cOff(d.free):d.cOff(d.free)+n], d.length[d.cOff(sample):d.cOff(sample)+n])
		copy(d.c[d.cOff(d.free):d.cOff(d.free)+n], d.c[d.cOff(sample):d.cOff(sample)+n])
		pos = d.free
		d.free = int(d.point[d.free])
		if d.free == 0 {
			return start, false
		}
	}
	d.point[pos] = 0
	pos = start
	for j := 1; j <= d.maxi; j++ {
		dim := d.arrayi[j] - 1
		d.c[d.cOff(pos)+dim] = d.c[d.cOff(sample)+dim] + delta
		pos = int(d.point[pos])
		d.c[d.cOff(pos)+dim] = d.c[d.cOff(sample)+dim] - delta
		pos = int(d.point[pos])
	}
	return start, true
}

// divide is DIRDivide: it orders the sampled dimensions by the smaller of
// their two values and shortens the sides of the new rectangles and of the
// divided one accordingly.
func (d *dirState) divide(new_, currentlength, sample int) {
	start := 0
	pos := new_
	for i := 1; i <= d.maxi; i++ {
		j := d.arrayi[i]
		d.w[j] = d.fv[pos]
		k := pos
		pos = int(d.point[pos])
		d.w[j] = dirMin(d.fv[pos], d.w[j])
		pos = int(d.point[pos])
		d.insertlist2(&start, j, k)
	}
	cl := int32(currentlength + 1)
	for j := 1; j <= d.maxi; j++ {
		// dirsearchmin_
		k := start
		pos = d.l2pos[start]
		start = d.l2next[start]
		pos2 := start
		d.length[d.cOff(sample)+k-1] = cl
		for i := 1; i <= d.maxi-j+1; i++ {
			d.length[d.cOff(pos)+k-1] = cl
			pos = int(d.point[pos])
			d.length[d.cOff(pos)+k-1] = cl
			if pos2 > 0 {
				pos = d.l2pos[pos2]
				pos2 = d.l2next[pos2]
			}
		}
	}
}

// getI is DIRGet_I: the dimensions whose side is maximal (smallest index).
func (d *dirState) getI(pos int) {
	l := d.length[d.cOff(pos) : d.cOff(pos)+d.n]
	help := l[0]
	for i := 1; i < len(l); i++ {
		if l[i] < help {
			help = l[i]
		}
	}
	j := 1
	for i := 1; i <= d.n; i++ {
		if l[i-1] == help {
			d.arrayi[j] = i
			j++
		}
	}
	d.maxi = j - 1
}

// samplef is DIRSamplef (with DIRinfcn): it evaluates the objective at the
// 2*maxi new points, in list order, and then updates fmax, the feasibility
// flags and the best point. With several workers the evaluations run
// concurrently; the bookkeeping is done afterwards in list order, so the
// result does not depend on the worker count.
func (d *dirState) samplef(new_ int) {
	cnt := 2 * d.maxi
	d.posBuf = d.posBuf[:0]
	pos := new_
	for j := 0; j < cnt; j++ {
		d.posBuf = append(d.posBuf, pos)
		pos = int(d.point[pos])
	}
	if cap(d.valBuf) < cnt {
		d.valBuf = make([]float64, cnt)
		d.flagBuf = make([]int, cnt)
	}
	vals, flags := d.valBuf[:cnt], d.flagBuf[:cnt]
	if d.workers > 1 && cnt > 1 {
		d.evalParallel(vals, flags)
	} else {
		for j, p := range d.posBuf {
			d.scalePoint(p, d.xbuf)
			vals[j], flags[j] = d.fn(d.xbuf)
		}
	}
	d.scalePoint(d.posBuf[cnt-1], d.xLast)
	for j, p := range d.posBuf {
		d.fv[p] = vals[j]
		kret := flags[j]
		if kret > d.iinfesible {
			d.iinfesible = kret
		}
		if kret == 0 {
			d.fk[p] = 0
			d.ifeasiblef = 0
			d.fmax = dirMax(d.fv[p], d.fmax)
		}
		if kret >= 1 {
			d.fk[p] = 2
			d.fv[p] = d.fmax
		}
	}
	pos = new_
	for j := 1; j <= cnt; j++ {
		if d.fv[pos] < d.minf && d.fk[pos] == 0 {
			d.minf = d.fv[pos]
			d.minpos = pos
		}
		pos = int(d.point[pos])
	}
}

// scalePoint writes the point of rectangle pos in the user's coordinates,
// (c + xs2) * xs1 as DIRinfcn does.
func (d *dirState) scalePoint(pos int, x []float64) {
	off := d.cOff(pos)
	for i := range x {
		x[i] = (d.c[off+i] + d.xs2[i]) * d.xs1[i]
	}
}
