package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"sort"
)

// Limits on what Decode accepts, so that a hostile or corrupted fixture
// cannot make the harness allocate without bound.
const (
	maxFixtureBytes = 64 << 20
	maxCases        = 10000
	maxVars         = 5000
	maxRows         = 5000
	// Infinity is written as +-1e30 in a fixture (JSON has no Inf); a bound of
	// magnitude >= infThreshold is infinite (OSQP_INFTY is 1e30, the
	// Maros–Mészáros loader uses 1e20).
	infThreshold = 1e20
)

// Fixture is the content of testdata/qp/<set>.json: a set of quadratic
// programs min 1/2 x'Px + q'x s.t. l <= Ax <= u with the results of the
// reference solvers, and the environment they ran in.
type Fixture struct {
	Osqp     string `json:"osqp"`
	OsqpCore string `json:"osqp_core"`
	Clarabel string `json:"clarabel"`
	Quadprog string `json:"quadprog"`
	Numpy    string `json:"numpy"`
	Scipy    string `json:"scipy"`
	Python   string `json:"python"`
	Machine  string `json:"machine"`
	System   string `json:"system"`
	Cases    []Case `json:"cases"`
}

// Case is one problem with its results, keyed by solver name ("osqp",
// "osqp_tight", "clarabel", "quadprog").
type Case struct {
	Name string      `json:"name"`
	Note string      `json:"note"`
	N    int         `json:"n"`
	P    [][]float64 `json:"p"` // n×n, full symmetric; null when PCSC is given
	Q    []float64   `json:"q"`
	A    [][]float64 `json:"a"` // m×n; null when ACSC is given
	// The sparse form, for problems whose dense matrices would be too large (the
	// Maros–Mészáros problems over the dense cap): P full symmetric n×n and A
	// m×n in compressed sparse column form. M is the number of rows of A.
	M       int               `json:"m,omitempty"`
	PCSC    *CSC              `json:"p_csc,omitempty"`
	ACSC    *CSC              `json:"a_csc,omitempty"`
	L       []float64         `json:"l"`
	U       []float64         `json:"u"`
	Results map[string]Result `json:"results"`
}

// Result is one solver's answer. X and Y are nil when the solver returned
// non-finite values (an infeasible or unbounded problem); Error is the
// exception the solver raised instead. Y is the solver's own multiplier
// convention (OSQP: P x + q + A'y = 0; Clarabel: the cone duals z).
type Result struct {
	Status     string    `json:"status,omitempty"`
	X          []float64 `json:"x,omitempty"`
	Y          []float64 `json:"y,omitempty"`
	Obj        *float64  `json:"obj,omitempty"`
	Iter       *int      `json:"iter,omitempty"`
	RhoUpdates *int      `json:"rho_updates,omitempty"`
	PrimRes    *float64  `json:"prim_res,omitempty"`
	DualRes    *float64  `json:"dual_res,omitempty"`
	Error      string    `json:"error,omitempty"`
}

// IsInf reports whether a bound written in a fixture stands for infinity.
func IsInf(v float64) bool { return math.Abs(v) >= infThreshold }

// Decode reads a fixture, rejecting unknown fields, trailing data, more than
// maxFixtureBytes, a fixture without cases, duplicate or empty names,
// dimensions that do not agree, and non-finite or inconsistent bounds.
func Decode(r io.Reader) (*Fixture, error) {
	lr := &io.LimitedReader{R: r, N: maxFixtureBytes + 1}
	dec := json.NewDecoder(lr)
	dec.DisallowUnknownFields()
	var f Fixture
	if err := dec.Decode(&f); err != nil {
		return nil, fmt.Errorf("fixture: %w", err)
	}
	if _, err := dec.Token(); err != io.EOF {
		return nil, fmt.Errorf("fixture: data after the top-level value")
	}
	if lr.N <= 0 { // the reader stopped at the limit: the input was longer
		return nil, fmt.Errorf("fixture: larger than %d bytes", maxFixtureBytes)
	}
	if len(f.Cases) == 0 || len(f.Cases) > maxCases {
		return nil, fmt.Errorf("fixture: %d cases (want 1 to %d)", len(f.Cases), maxCases)
	}
	seen := map[string]bool{}
	for i := range f.Cases {
		c := &f.Cases[i]
		if c.Name == "" || seen[c.Name] {
			return nil, fmt.Errorf("fixture: case %d has an empty or duplicate name %q", i, c.Name)
		}
		seen[c.Name] = true
		if err := c.validate(); err != nil {
			return nil, fmt.Errorf("fixture: case %q: %w", c.Name, err)
		}
	}
	return &f, nil
}

// CSC is a matrix in compressed sparse column form: column j holds the entries
// Indices[Indptr[j]:Indptr[j+1]] (strictly increasing rows) with Data.
type CSC struct {
	Indptr  []int     `json:"indptr"`
	Indices []int     `json:"indices"`
	Data    []float64 `json:"data"`
}

// maxNNZ bounds the entries of a sparse matrix a fixture may hold.
const maxNNZ = 4 << 20

// validate checks a sparse matrix of the given shape: pointer array of length
// cols+1 starting at 0 and non-decreasing, equal numbers of indices and data,
// rows in range and strictly increasing within a column.
func (s *CSC) validate(rows, cols int) error {
	if len(s.Indptr) != cols+1 || s.Indptr[0] != 0 {
		return fmt.Errorf("indptr has %d entries (want %d, starting at 0)", len(s.Indptr), cols+1)
	}
	nnz := s.Indptr[cols]
	if nnz < 0 || nnz > maxNNZ || len(s.Indices) != nnz || len(s.Data) != nnz {
		return fmt.Errorf("%d entries announced, %d indices, %d data (limit %d)", nnz, len(s.Indices), len(s.Data), maxNNZ)
	}
	for j := 0; j < cols; j++ {
		lo, hi := s.Indptr[j], s.Indptr[j+1]
		if lo > hi || hi > nnz {
			return fmt.Errorf("indptr decreases or overruns at column %d", j)
		}
		for k := lo; k < hi; k++ {
			if s.Indices[k] < 0 || s.Indices[k] >= rows || (k > lo && s.Indices[k] <= s.Indices[k-1]) {
				return fmt.Errorf("column %d: row %d out of range or not increasing", j, s.Indices[k])
			}
			if math.IsNaN(s.Data[k]) || math.IsInf(s.Data[k], 0) {
				return fmt.Errorf("column %d: non-finite entry", j)
			}
		}
	}
	return nil
}

// Rows returns the number of rows of A in either form.
func (c *Case) Rows() int {
	if c.ACSC != nil {
		return c.M
	}
	return len(c.A)
}

func (c *Case) validate() error {
	if c.N < 1 || c.N > maxVars {
		return fmt.Errorf("n = %d (want 1 to %d)", c.N, maxVars)
	}
	if c.PCSC != nil && c.P != nil || c.ACSC != nil && c.A != nil {
		return fmt.Errorf("a matrix is given both dense and sparse")
	}
	if c.ACSC == nil && c.M != 0 {
		return fmt.Errorf("m = %d without a sparse a", c.M)
	}
	m := c.Rows()
	if m < 0 || m > maxRows || len(c.L) != m || len(c.U) != m || len(c.Q) != c.N || (c.PCSC == nil && len(c.P) != c.N) {
		return fmt.Errorf("dimensions disagree (n %d, q %d, p %d rows, a %d rows, l %d, u %d)",
			c.N, len(c.Q), len(c.P), m, len(c.L), len(c.U))
	}
	if c.PCSC != nil {
		if err := c.PCSC.validate(c.N, c.N); err != nil {
			return fmt.Errorf("p_csc: %w", err)
		}
	}
	if c.ACSC != nil {
		if err := c.ACSC.validate(m, c.N); err != nil {
			return fmt.Errorf("a_csc: %w", err)
		}
	}
	for i, row := range c.P {
		if len(row) != c.N {
			return fmt.Errorf("p row %d has %d entries, want %d", i, len(row), c.N)
		}
	}
	for i, row := range c.A {
		if len(row) != c.N {
			return fmt.Errorf("a row %d has %d entries, want %d", i, len(row), c.N)
		}
	}
	for i := range c.L {
		if c.L[i] > c.U[i] || (IsInf(c.L[i]) && c.L[i] > 0) || (IsInf(c.U[i]) && c.U[i] < 0) {
			return fmt.Errorf("row %d has the bounds [%g, %g]", i, c.L[i], c.U[i])
		}
	}
	for name, r := range c.Results {
		if name == "" {
			return fmt.Errorf("a result without a solver name")
		}
		if r.X != nil && len(r.X) != c.N {
			return fmt.Errorf("%s: x has %d entries, want %d", name, len(r.X), c.N)
		}
		if r.Y != nil && len(r.Y) > maxRows+2*maxRows {
			return fmt.Errorf("%s: y has %d entries", name, len(r.Y))
		}
	}
	return nil
}

// Encode writes a fixture as indented JSON.
func Encode(w io.Writer, f *Fixture) error {
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetIndent("", " ")
	if err := enc.Encode(f); err != nil {
		return err
	}
	_, err := w.Write(buf.Bytes())
	return err
}

// Disagreement is one difference between two fixtures.
type Disagreement struct {
	Case, Solver, What string
}

func (d Disagreement) String() string {
	if d.Solver == "" {
		return fmt.Sprintf("%s: %s", d.Case, d.What)
	}
	return fmt.Sprintf("%s [%s]: %s", d.Case, d.Solver, d.What)
}

func relClose(a, b, tol float64) bool {
	return math.Abs(a-b) <= tol*math.Max(1, math.Max(math.Abs(a), math.Abs(b)))
}

func vecClose(a, b []float64, tol float64) bool {
	if (a == nil) != (b == nil) || len(a) != len(b) {
		return false
	}
	for i := range a {
		if !relClose(a[i], b[i], tol) {
			return false
		}
	}
	return true
}

// Diff lists where two fixtures disagree: cases present on one side only,
// different problem data (bit for bit), and, for each solver and case, the
// status, the iteration count (exactly) and x, y and the objective (to the
// relative tolerance tol).
func Diff(a, b *Fixture, tol float64) []Disagreement {
	var out []Disagreement
	bi := map[string]*Case{}
	for i := range b.Cases {
		bi[b.Cases[i].Name] = &b.Cases[i]
	}
	inA := map[string]bool{}
	for i := range a.Cases {
		ca := &a.Cases[i]
		inA[ca.Name] = true
		cb, ok := bi[ca.Name]
		if !ok {
			out = append(out, Disagreement{ca.Name, "", "only in the first fixture"})
			continue
		}
		if !sameProblem(ca, cb) {
			out = append(out, Disagreement{ca.Name, "", "the problem data differ"})
			continue
		}
		names := map[string]bool{}
		for s := range ca.Results {
			names[s] = true
		}
		for s := range cb.Results {
			names[s] = true
		}
		sorted := make([]string, 0, len(names))
		for s := range names {
			sorted = append(sorted, s)
		}
		sort.Strings(sorted)
		for _, s := range sorted {
			ra, oka := ca.Results[s]
			rb, okb := cb.Results[s]
			for _, what := range diffResult(ra, rb, oka, okb, tol) {
				out = append(out, Disagreement{ca.Name, s, what})
			}
		}
	}
	for i := range b.Cases {
		if !inA[b.Cases[i].Name] {
			out = append(out, Disagreement{b.Cases[i].Name, "", "only in the second fixture"})
		}
	}
	return out
}

func sameCSC(a, b *CSC) bool {
	if a == nil || b == nil {
		return a == b
	}
	if len(a.Indptr) != len(b.Indptr) || len(a.Indices) != len(b.Indices) {
		return false
	}
	for i := range a.Indptr {
		if a.Indptr[i] != b.Indptr[i] {
			return false
		}
	}
	for i := range a.Indices {
		if a.Indices[i] != b.Indices[i] {
			return false
		}
	}
	return vecClose(a.Data, b.Data, 0)
}

func sameProblem(a, b *Case) bool {
	if a.N != b.N || !vecClose(a.Q, b.Q, 0) || !vecClose(a.L, b.L, 0) || !vecClose(a.U, b.U, 0) ||
		len(a.P) != len(b.P) || len(a.A) != len(b.A) || a.M != b.M || !sameCSC(a.PCSC, b.PCSC) || !sameCSC(a.ACSC, b.ACSC) {
		return false
	}
	for i := range a.P {
		if !vecClose(a.P[i], b.P[i], 0) {
			return false
		}
	}
	for i := range a.A {
		if !vecClose(a.A[i], b.A[i], 0) {
			return false
		}
	}
	return true
}

func diffResult(a, b Result, oka, okb bool, tol float64) []string {
	switch {
	case !oka:
		return []string{"only in the second fixture"}
	case !okb:
		return []string{"only in the first fixture"}
	}
	var out []string
	if a.Status != b.Status {
		out = append(out, fmt.Sprintf("status %q against %q", a.Status, b.Status))
	}
	if a.Error != b.Error {
		out = append(out, fmt.Sprintf("error %q against %q", a.Error, b.Error))
	}
	if !vecClose(a.X, b.X, tol) {
		out = append(out, "x differs")
	}
	if !vecClose(a.Y, b.Y, tol) {
		out = append(out, "y differs")
	}
	if (a.Obj == nil) != (b.Obj == nil) || (a.Obj != nil && !relClose(*a.Obj, *b.Obj, tol)) {
		out = append(out, "objective differs")
	}
	if (a.Iter == nil) != (b.Iter == nil) || (a.Iter != nil && *a.Iter != *b.Iter) {
		out = append(out, "iteration count differs")
	}
	return out
}
