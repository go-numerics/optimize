package optimize

import (
	"fmt"
	"math"
)

// Updates and warm starts of a QPSolver (design-qp.md §5). For the active set
// the factors J, R and the working set survive UpdateQ and UpdateBounds: the
// next Solve recovers x and the multipliers from them and drops the
// constraints whose multiplier went negative (Schmelzer & Stoll, Prop. 8.1–8.2)
// before the walk resumes. UpdateMatrixValues changes G, so the factors are
// rebuilt, entering the previous active set as the hint. Whatever the path, the
// answer is the one a cold solve gives (Remark 8.1); only Nit differs, because
// it counts the working-set edits actually made.

// UpdateQ replaces the linear term; every factor is kept.
func (s *QPSolver) UpdateQ(q []float64) error {
	if len(q) != s.m.n {
		return fmt.Errorf("%w: q has %d entries, want %d", ErrDimension, len(q), s.m.n)
	}
	if !allFinite(q) {
		return fmt.Errorf("%w: q contains NaN or Inf", ErrInput)
	}
	copy(s.m.q, q)
	if s.ad != nil {
		s.ad.setQ(q) // the iterates stay as the warm start
	}
	return nil
}

// UpdateBounds replaces the sides l and u of the canonical rows (design-qp.md
// §2.2: AEq rows, AUb rows, the Linear blocks, the ASparse rows, then the finite
// bounds; one entry per canonical row, -Inf and +Inf for a missing side). A row
// keeps the kind it was created with: a finite side cannot become infinite or the
// reverse, and an equality (l == u) cannot become an interval or the reverse;
// otherwise the error is ErrBounds. The active-set factors are kept. ADMM takes
// any sides and refactors only if a row changes type (OSQP's update_rho_vec).
func (s *QPSolver) UpdateBounds(l, u []float64) error {
	m := s.m
	if len(l) != len(m.rows) || len(u) != len(m.rows) {
		return fmt.Errorf("%w: l and u have %d and %d entries, want %d canonical rows", ErrDimension, len(l), len(u), len(m.rows))
	}
	if err := checkSides(l, u, "canonical row"); err != nil {
		return err
	}
	for i, r := range m.rows {
		// The active set keeps the kind of each row (its columns); ADMM takes any
		// sides: a row whose type (loose, inequality, equality) changes gets a new
		// ρ and a numeric refactorisation.
		if s.ad == nil && (math.IsInf(l[i], -1) != math.IsInf(r.l, -1) || math.IsInf(u[i], 1) != math.IsInf(r.u, 1) || (l[i] == u[i]) != (r.l == r.u)) {
			return fmt.Errorf("%w: canonical row %d changes kind (finite sides or equality status)", ErrBounds, i)
		}
	}
	for i := range m.rows {
		m.rows[i].l, m.rows[i].u = l[i], u[i]
	}
	if s.ad != nil {
		s.ad.setBounds()
	}
	if g := s.gi; g != nil {
		for i, c := range g.cols {
			if r := m.rows[c.row]; c.sign > 0 {
				g.cols[i].b = r.l
			} else {
				g.cols[i].b = -r.u
			}
		}
	}
	return nil
}

// UpdateMatrixValues replaces the values of P and of the dense constraint rows
// without changing their patterns. p holds the n×n values row-major for a dense
// P (symmetrised as (P+Pᵀ)/2), or the entries of PSparse in their stored order;
// a holds the dense canonical rows (AEq, AUb, the Linear blocks) row-major, the
// bound rows excluded, followed by the entries of ASparse in its CSC storage
// order (ADMM only). nil keeps the old values. The active-set factors are
// rebuilt on the next Solve, seeded with the previous active set; a P that is no
// longer positive definite is rejected and nothing changes. The ADMM solver
// reruns its scaling and refactors numerically on the stored symbolic factor
// (a new ordering only if the nonzero pattern of a dense P changed), keeps ρ
// and the iterates as the warm start, and rejects a P that makes the KKT matrix
// not quasi-definite, leaving the solver as it was.
func (s *QPSolver) UpdateMatrixValues(p, a []float64) error {
	m := s.m
	n := m.n
	if p != nil {
		want := 0 // P = 0 has no values
		switch {
		case m.p != nil:
			want = n * n
		case m.pUpper != nil:
			want = m.pUpper.NNZ()
		}
		if len(p) != want {
			return fmt.Errorf("%w: p has %d values, want %d", ErrDimension, len(p), want)
		}
		if !allFinite(p) {
			return fmt.Errorf("%w: p contains NaN or Inf", ErrInput)
		}
	}
	spN := 0 // entries of ASparse: the values after the dense rows' (ADMM only)
	if m.sp != nil {
		spN = m.sp.NNZ()
	}
	if a != nil {
		if len(a) != len(m.dense)+spN {
			return fmt.Errorf("%w: a has %d values, want %d", ErrDimension, len(a), len(m.dense)+spN)
		}
		if !allFinite(a) {
			return fmt.Errorf("%w: a contains NaN or Inf", ErrInput)
		}
	}
	oldP, oldD, oldC, oldSP := m.p, m.dense, m.chol, m.sp
	var oldU []float64
	if m.pUpper != nil {
		oldU = m.pUpper.Data
	}
	switch {
	case p != nil && m.p != nil:
		d := clone(p)
		for i := 0; i < n; i++ {
			for j := 0; j < i; j++ {
				v := 0.5 * (d[i*n+j] + d[j*n+i])
				d[i*n+j], d[j*n+i] = v, v
			}
		}
		m.p = d
	case p != nil:
		m.pUpper.Data = clone(p)
	}
	if a != nil {
		m.dense = clone(a[:len(m.dense)])
		if spN > 0 { // same pattern: CSR → CSC, new values, back to CSR
			c := m.sp.ToCSC()
			copy(c.Data, a[len(m.dense):])
			m.sp = c.ToCSR()
		}
	}
	restore := func() {
		m.p, m.dense, m.chol, m.sp = oldP, oldD, oldC, oldSP
		if oldU != nil {
			m.pUpper.Data = oldU
		}
	}
	if s.ad != nil {
		err := s.ad.resetMatrices()
		if err == nil && s.ad.nonConvex != nil {
			err = fmt.Errorf("%w: the updated KKT matrix is not quasi-definite: %v", ErrInput, s.ad.nonConvex)
		}
		if err != nil { // back to the data that worked
			restore()
			// The old pattern is the one the stored symbolic factor was made for,
			// so this cannot need a new ordering and cannot fail.
			_ = s.ad.resetMatrices()
			return err
		}
		return nil
	}
	if p != nil && !m.positiveDefinite() {
		restore()
		return fmt.Errorf("%w: the updated P is not positive definite; the active-set method needs it", ErrInput)
	}
	s.gi = nil
	return nil
}

// WarmStart sets the starting point of the next Solve: x (length n) and y (the
// canonical rows' multipliers), either may be nil. The active set ignores them
// (it starts from its working set, design-qp.md §5); they are validated all the
// same, and kept for the ADMM solver.
func (s *QPSolver) WarmStart(x, y []float64) error {
	if x != nil && len(x) != s.m.n {
		return fmt.Errorf("%w: x has %d entries, want %d", ErrDimension, len(x), s.m.n)
	}
	if y != nil && len(y) != len(s.m.rows) {
		return fmt.Errorf("%w: y has %d entries, want %d (canonical rows)", ErrDimension, len(y), len(s.m.rows))
	}
	if !allFinite(x) || !allFinite(y) {
		return fmt.Errorf("%w: x and y must be finite", ErrInput)
	}
	if s.ad != nil {
		s.ad.setWarm(x, y)
	}
	return nil
}
