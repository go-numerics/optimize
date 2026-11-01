package optimize

import (
	"fmt"
	"math"

	"github.com/go-simd/floats"
)

// deSolver is scipy's DifferentialEvolutionSolver. The population lives in the
// unit cube, one row per member; rows are swapped by pointer when the best
// member is promoted.
type deSolver struct {
	f   func([]float64) float64
	o   *DEOptions
	rng *RandomSource

	n, s            int
	lo, hi          []float64
	arg1, arg2      []float64 // scale arguments: centre and width
	recip           []float64 // 1/width, 0 for a zero width
	strat           deStrategyInfo
	scale           float64
	dither          bool
	d0, d1          float64
	cr, tol, atol   float64
	maxiter, maxfev int
	deferred        bool
	workers         int

	nfev     int
	pop      [][]float64
	flat     []float64
	energies []float64
	idx      []int64 // persistent permutation of 0..S-1 (never reset)

	// scratch, sized once
	trial    []float64
	bprime   []float64
	x        []float64
	cross    []bool
	oobBuf   []float64
	smp      []int64 // 5 samples
	smpAll   [][]int64
	trials   [][]float64
	trialsX  [][]float64
	bprimes  [][]float64
	tEnergy  []float64
	fill     []int64
	crossAll []float64
	accept   []bool
	scaled   [][]float64 // custom strategy: the population in problem units
	dev      []float64
}

// alloc sizes the population and the scratch space for s.s members.
func (s *deSolver) alloc() {
	s.flat = make([]float64, s.s*s.n)
	s.pop = make([][]float64, s.s)
	for i := range s.pop {
		s.pop[i] = s.flat[i*s.n : (i+1)*s.n : (i+1)*s.n]
	}
	s.energies = make([]float64, s.s)
	for i := range s.energies {
		s.energies[i] = math.Inf(1)
	}
	s.idx = make([]int64, s.s)
	for i := range s.idx {
		s.idx[i] = int64(i)
	}
	s.dev = make([]float64, s.s)
	s.trial = make([]float64, s.n)
	s.bprime = make([]float64, s.n)
	s.x = make([]float64, s.n)
	s.cross = make([]bool, s.n)
	s.oobBuf = make([]float64, s.n)
	s.smp = make([]int64, 5)
	if s.deferred {
		s.allocDeferred()
	}
	if s.o.CustomStrategy != nil {
		s.scaled = matrix(s.s, s.n)
	}
}

func matrix(r, c int) [][]float64 {
	flat := make([]float64, r*c)
	m := make([][]float64, r)
	for i := range m {
		m[i] = flat[i*c : (i+1)*c : (i+1)*c]
	}
	return m
}

func (s *deSolver) allocDeferred() {
	s.trials = matrix(s.s, s.n)
	s.trialsX = matrix(s.s, s.n)
	s.bprimes = matrix(s.s, s.n)
	s.smpAll = make([][]int64, s.s)
	for i := range s.smpAll {
		s.smpAll[i] = make([]int64, 5)
	}
	s.tEnergy = make([]float64, s.s)
	s.fill = make([]int64, s.s)
	s.crossAll = make([]float64, s.s*s.n)
	s.accept = make([]bool, s.s)
}

// scaleInto is _scale_parameters: x = m + (t-0.5)·w. (Integrality, task 6.)
func (s *deSolver) scaleInto(dst, t []float64) {
	for j, v := range t {
		dst[j] = s.arg1[j] + float64((v-0.5)*s.arg2[j])
	}
}

// unscale is _unscale_parameters: t = (x-m)·r + 0.5.
func (s *deSolver) unscale(dst, x []float64) {
	for j, v := range x {
		dst[j] = float64((v-s.arg1[j])*s.recip[j]) + 0.5
	}
}

// initLHS is init_population_lhs.
func (s *deSolver) initLHS() {
	s.rng.fillUnit(s.flat)
	seg := 1.0 / float64(s.s)
	samples := make([]float64, len(s.flat))
	for i := range s.s {
		y := float64(i)*seg + 0.0 // linspace(0, 1, S, endpoint=False)
		for j := range s.n {
			samples[i*s.n+j] = float64(seg*s.flat[i*s.n+j]) + y
		}
	}
	order := make([]int64, s.s)
	for j := range s.n {
		for i := range order {
			order[i] = int64(i)
		}
		s.rng.ShuffleInt64(order)
		for i := range s.s {
			s.pop[i][j] = samples[int(order[i])*s.n+j]
		}
	}
}

// argmin is numpy's argmin: the first minimum, the first NaN winning.
func argmin(e []float64) int {
	best := 0
	for i, v := range e {
		if math.IsNaN(v) {
			return i
		}
		if v < e[best] {
			best = i
		}
	}
	return best
}

// promote is _promote_lowest_energy.
func (s *deSolver) promote() {
	l := argmin(s.energies)
	s.energies[0], s.energies[l] = s.energies[l], s.energies[0]
	s.pop[0], s.pop[l] = s.pop[l], s.pop[0]
}

func allInf(e []float64) bool {
	for _, v := range e {
		if !math.IsInf(v, 0) {
			return false
		}
	}
	return true
}

func anyInf(e []float64) bool {
	for _, v := range e {
		if math.IsInf(v, 0) {
			return true
		}
	}
	return false
}

// initEnergies evaluates the whole population (the first block of solve and
// of __next__), within the evaluation budget.
func (s *deSolver) initEnergies() {
	s.evalPopulation(s.pop, s.energies)
	s.promote()
}

// evalPopulation is _calculate_population_energies for the members in
// members: the first min(len, maxfev-nfev) are evaluated, the others get +Inf.
func (s *deSolver) evalPopulation(members [][]float64, out []float64) {
	k := min(len(members), s.maxfev-s.nfev)
	k = max(k, 0)
	pts := s.trialsX
	if pts == nil || len(pts) < k {
		pts = matrix(len(members), s.n)
	}
	for i := range k {
		s.scaleInto(pts[i], members[i])
	}
	for i := range members {
		out[i] = math.Inf(1)
	}
	evalBatch(s.f, pts[:k], out[:k], s.workers)
	s.nfev += k
}

// The population arithmetic is plain Go in NumPy's operation order, one
// small function per operation, each product rounded before it is added so
// that no fused multiply-add can change a bit (SciPy's wheels do not fuse).

// deAxDiff: dst = a + F·(b − c).
func deAxDiff(dst, a []float64, f float64, b, c []float64) {
	for j := range dst {
		dst[j] = a[j] + float64(f*(b[j]-c[j]))
	}
}

// deRelax: dst += F·(a − dst).
func deRelax(dst []float64, f float64, a []float64) {
	for j := range dst {
		dst[j] = dst[j] + float64(f*(a[j]-dst[j]))
	}
}

// deAddDiff: dst += F·(a − b).
func deAddDiff(dst []float64, f float64, a, b []float64) {
	for j := range dst {
		dst[j] = dst[j] + float64(f*(a[j]-b[j]))
	}
}

// deCurrentToBest: dst = c + F·(((best − c) + r0) − r1).
func deCurrentToBest(dst, c []float64, f float64, best, r0, r1 []float64) {
	for j := range dst {
		dst[j] = c[j] + float64(f*(((best[j]-c[j])+r0[j])-r1[j]))
	}
}

// deAx4: dst = a + F·(((b + c) − d) − e).
func deAx4(dst, a []float64, f float64, b, c, d, e []float64) {
	for j := range dst {
		dst[j] = a[j] + float64(f*(((b[j]+c[j])-d[j])-e[j]))
	}
}

// selectSamples is _select_samples: shuffle the persistent index array,
// take idx[:6], drop the candidate, keep the first five. It returns the
// number of samples written to out.
func (s *deSolver) selectSamples(c int, out []int64) int {
	s.rng.ShuffleInt64(s.idx)
	k := 0
	for _, v := range s.idx[:min(6, s.s)] {
		if v != int64(c) && k < 5 {
			out[k] = v
			k++
		}
	}
	return k
}

// bprimeInto computes the mutant of candidate c from the samples.
func (s *deSolver) bprimeInto(dst []float64, c int, smp []int64) {
	p := s.pop
	f := s.scale
	r := func(i int) []float64 { return p[smp[i]] }
	switch s.strat.kind {
	case deBest1:
		deAxDiff(dst, p[0], f, r(0), r(1))
	case deRand1:
		deAxDiff(dst, r(0), f, r(1), r(2))
	case deRandToBest1:
		copy(dst, r(0))
		deRelax(dst, f, p[0])
		deAddDiff(dst, f, r(1), r(2))
	case deCurrentToBest1:
		deCurrentToBest(dst, p[c], f, p[0], r(0), r(1))
	case deBest2:
		deAx4(dst, p[0], f, r(0), r(1), r(2), r(3))
	default: // deRand2
		deAx4(dst, r(0), f, r(1), r(2), r(3), r(4))
	}
}

// crossover builds the trial from the current member c and the mutant, with
// cross[j] already holding (u_j < CR), starting at fill.
func (s *deSolver) crossover(trial, cur, bprime []float64, cross []bool, fill int) {
	copy(trial, cur)
	if !s.strat.exp {
		cross[fill] = true
		for j, ok := range cross {
			if ok {
				trial[j] = bprime[j]
			}
		}
		return
	}
	cross[0] = true // index 0, not fill: SciPy's behaviour
	k := fill
	for i := 0; i < s.n && cross[i]; i++ {
		trial[k] = bprime[k]
		k = (k + 1) % s.n
	}
}

// ensureBounds is _ensure_constraint: out-of-cube components are redrawn
// with one uniform call, in index order.
func (s *deSolver) ensureBounds(t []float64) {
	oob := 0
	for _, v := range t {
		if v > 1 || v < 0 {
			oob++
		}
	}
	if oob == 0 {
		return
	}
	buf := s.oobBuf[:oob]
	s.rng.fillUnit(buf)
	k := 0
	for j, v := range t {
		if v > 1 || v < 0 {
			t[j] = buf[k]
			k++
		}
	}
}

// customTrial runs the user strategy for candidate c on the population in
// problem units and unscales the result into dst.
func (s *deSolver) customTrial(dst []float64, c int) {
	for i := range s.pop {
		s.scaleInto(s.scaled[i], s.pop[i])
	}
	s.customOne(dst, c)
}

func (s *deSolver) customOne(dst []float64, c int) {
	s.o.CustomStrategy(c, s.scaled, s.rng, s.x)
	s.unscale(dst, s.x)
}

// generation is __next__ for one generation. It returns false for
// StopIteration (the evaluation budget).
func (s *deSolver) generation() bool {
	if allInf(s.energies) {
		s.initEnergies()
	}
	if s.dither {
		s.scale = s.rng.Uniform(s.d0, s.d1)
	}
	if s.deferred {
		return s.generationDeferred()
	}
	return s.generationImmediate()
}

func (s *deSolver) generationImmediate() bool {
	for c := range s.s {
		if s.nfev > s.maxfev {
			return false
		}
		trial := s.trial
		if s.o.CustomStrategy != nil {
			s.customTrial(trial, c)
		} else {
			fill := int(s.rng.IntN(int64(s.n)))
			k := s.selectSamples(c, s.smp)
			s.bprimeInto(s.bprime, c, s.smp[:k])
			s.rng.fillUnit(s.oobBuf)
			for j := range s.n {
				s.cross[j] = s.oobBuf[j] < s.cr
			}
			s.crossover(trial, s.pop[c], s.bprime, s.cross, fill)
		}
		s.ensureBounds(trial)
		s.scaleInto(s.x, trial)
		energy := s.f(s.x)
		s.nfev++
		if energy <= s.energies[c] {
			copy(s.pop[c], trial)
			s.energies[c] = energy
			if energy <= s.energies[0] {
				s.promote()
			}
		}
	}
	return true
}

func (s *deSolver) generationDeferred() bool {
	if s.nfev >= s.maxfev {
		return false
	}
	n, sz := s.n, s.s
	// mutate_many: shuffles for all candidates first
	if s.o.CustomStrategy != nil {
		for i := range s.pop {
			s.scaleInto(s.scaled[i], s.pop[i])
		}
		for c := range sz {
			s.customOne(s.trials[c], c)
		}
	} else {
		for c := range sz {
			s.smpAll[c] = s.smpAll[c][:5]
			k := s.selectSamples(c, s.smpAll[c])
			s.smpAll[c] = s.smpAll[c][:k]
		}
		for c := range sz {
			s.bprimeInto(s.bprimes[c], c, s.smpAll[c])
		}
		s.rng.fillBelow(s.fill, int64(n))
		s.rng.fillUnit(s.crossAll)
		for c := range sz {
			for j := range n {
				s.cross[j] = s.crossAll[c*n+j] < s.cr
			}
			s.crossover(s.trials[c], s.pop[c], s.bprimes[c], s.cross, int(s.fill[c]))
		}
	}
	// ensure_constraint over the S×N block, row-major
	oob := 0
	for _, row := range s.trials {
		for _, v := range row {
			if v > 1 || v < 0 {
				oob++
			}
		}
	}
	if oob > 0 {
		buf := s.crossAll[:oob] // the crossover draws are consumed
		s.rng.fillUnit(buf)
		k := 0
		for _, row := range s.trials {
			for j, v := range row {
				if v > 1 || v < 0 {
					row[j] = buf[k]
					k++
				}
			}
		}
	}
	s.evalPopulation(s.trials, s.tEnergy)
	for i := range sz {
		s.accept[i] = s.tEnergy[i] <= s.energies[i]
	}
	for i := range sz {
		if s.accept[i] {
			copy(s.pop[i], s.trials[i])
			s.energies[i] = s.tEnergy[i]
		}
	}
	s.promote()
	return true
}

// std and mean of the energies, in NumPy's order of operations.
func (s *deSolver) meanStd() (mean, std float64) {
	e := s.energies
	n := float64(len(e))
	mean = floats.SumPairwise(e) / n
	for i, v := range e {
		d := v - mean
		s.dev[i] = d * d
	}
	return mean, math.Sqrt(floats.SumPairwise(s.dev) / n)
}

func (s *deSolver) converged() bool {
	if anyInf(s.energies) {
		return false
	}
	mean, std := s.meanStd()
	return std <= s.atol+float64(s.tol*math.Abs(mean))
}

func (s *deSolver) convergence() float64 {
	if anyInf(s.energies) {
		return math.Inf(1)
	}
	mean, std := s.meanStd()
	return std / (math.Abs(mean) + epsmch)
}

// result is _result: an intermediate OptimizeResult.
func (s *deSolver) result(nit int, msg string, status int) *DEResult {
	r := &DEResult{
		X:                  make([]float64, s.n),
		Fun:                s.energies[0],
		Nfev:               s.nfev,
		Nit:                nit,
		Message:            msg,
		Status:             status,
		Success:            status == DEConverged,
		Population:         matrix(s.s, s.n),
		PopulationEnergies: clone(s.energies),
	}
	s.scaleInto(r.X, s.pop[0])
	for i := range s.pop {
		s.scaleInto(r.Population[i], s.pop[i])
	}
	return r
}

// solve is DifferentialEvolutionSolver.solve.
func (s *deSolver) solve() (*DEResult, error) {
	if allInf(s.energies) {
		s.initEnergies()
	}
	nit, status := 0, DEConverged
	msg := deMsgSuccess
	broke := false
	for nit = 1; nit <= s.maxiter; nit++ {
		if !s.generation() {
			status = DEMaxFev
			switch {
			case s.nfev > s.maxfev:
				msg = deMsgMaxFev
			case s.nfev == s.maxfev:
				msg = deMsgReached
			}
			broke = true
			break
		}
		if s.o.Callback != nil {
			res := s.result(nit, "in progress", DEConverged)
			res.Convergence = s.tol / (s.convergence() + epsmch)
			if s.o.Callback(res) {
				status, msg = DEStopped, deMsgStopped
			}
		}
		if status != DEConverged || s.converged() {
			broke = true
			break
		}
	}
	if !broke {
		nit = s.maxiter
		msg, status = deMsgMaxIter, DEMaxIter
	}
	res := s.result(nit, msg, status)
	if err := s.polish(res); err != nil {
		return nil, err
	}
	return res, nil
}

// polish is the polishing block of solve.
func (s *deSolver) polish(res *DEResult) error {
	if s.o.NoPolish {
		return nil
	}
	bounds := make([]Bound, s.n)
	for i := range bounds {
		bounds[i] = Bound{s.lo[i], s.hi[i]}
	}
	var (
		r   *Result
		err error
	)
	if s.o.Polish != nil {
		r, err = s.o.Polish(s.f, clone(res.X), bounds)
	} else {
		r, err = Minimize(Problem{Func: s.f}, clone(res.X), LBFGSB, &Options{Bounds: bounds})
	}
	if err != nil {
		return err
	}
	if r == nil || len(r.X) != s.n {
		return fmt.Errorf("%w: the polisher must return a result holding %d values", ErrInput, s.n)
	}
	s.nfev += r.Nfev
	res.Nfev = s.nfev
	if r.Fun < res.Fun && r.Success && within(r.X, s.lo, s.hi) {
		res.Fun = r.Fun
		res.X = clone(r.X)
		res.Jac = clone(r.Jac)
		s.energies[0] = r.Fun
		s.unscale(s.pop[0], r.X)
		// SciPy's result.population was scaled before polishing and keeps
		// the old best member; population_energies is the solver's own
		// array, so it sees the polished value.
		res.PopulationEnergies[0] = r.Fun
	}
	return nil
}

func within(x, lo, hi []float64) bool {
	for i, v := range x {
		if !(v <= hi[i]) || !(lo[i] <= v) {
			return false
		}
	}
	return true
}
