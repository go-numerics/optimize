package main

import (
	"encoding/json"
	"fmt"
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"
	"sort"
)

// CSC is a matrix in compressed sparse column form (rows strictly increasing in
// each column), the layout of the instance files.
type CSC struct {
	Rows   int       `json:"rows"`
	Cols   int       `json:"cols"`
	Indptr []int     `json:"indptr"`
	Index  []int     `json:"indices"`
	Data   []float64 `json:"data"`
}

// Instance is one quadratic program min 1/2 x'Px + q'x s.t. l <= Ax <= u. P is
// the upper triangle only (OSQP's convention); infinite bounds are written ±1e30.
type Instance struct {
	Name  string    `json:"name"`
	Class string    `json:"class"`
	N     int       `json:"n"`
	M     int       `json:"m"`
	P     CSC       `json:"p"`
	Q     []float64 `json:"q"`
	A     CSC       `json:"a"`
	L     []float64 `json:"l"`
	U     []float64 `json:"u"`
}

const infBound = 1e30

// coo is a triplet matrix under construction.
type coo struct {
	rows, cols int
	i, j       []int
	v          []float64
}

func newCOO(rows, cols int) *coo { return &coo{rows: rows, cols: cols} }

func (c *coo) add(i, j int, v float64) {
	if v != 0 {
		c.i, c.j, c.v = append(c.i, i), append(c.j, j), append(c.v, v)
	}
}

// block adds b with its top-left corner at (r0, c0).
func (c *coo) block(r0, c0 int, b *coo) {
	for k := range b.v {
		c.add(r0+b.i[k], c0+b.j[k], b.v[k])
	}
}

// scaled returns b times s.
func (b *coo) scaled(s float64) *coo {
	o := newCOO(b.rows, b.cols)
	for k := range b.v {
		o.add(b.i[k], b.j[k], s*b.v[k])
	}
	return o
}

func eye(n int) *coo {
	e := newCOO(n, n)
	for k := 0; k < n; k++ {
		e.add(k, k, 1)
	}
	return e
}

// csc converts to CSC, summing duplicates; upper keeps row <= column.
func (c *coo) csc(upper bool) CSC {
	type key struct{ j, i int }
	sum := map[key]float64{}
	for k := range c.v {
		if upper && c.i[k] > c.j[k] {
			continue
		}
		sum[key{c.j[k], c.i[k]}] += c.v[k]
	}
	keys := make([]key, 0, len(sum))
	for k, v := range sum {
		if v != 0 {
			keys = append(keys, k)
		}
	}
	sort.Slice(keys, func(a, b int) bool {
		if keys[a].j != keys[b].j {
			return keys[a].j < keys[b].j
		}
		return keys[a].i < keys[b].i
	})
	m := CSC{Rows: c.rows, Cols: c.cols, Indptr: make([]int, c.cols+1)}
	for _, k := range keys {
		m.Indptr[k.j+1]++
		m.Index = append(m.Index, k.i)
		m.Data = append(m.Data, sum[k])
	}
	for j := 0; j < c.cols; j++ {
		m.Indptr[j+1] += m.Indptr[j]
	}
	if m.Index == nil {
		m.Index, m.Data = []int{}, []float64{}
	}
	return m
}

type gen struct{ r *rand.Rand }

func newGen(seed uint64) *gen { return &gen{rand.New(rand.NewPCG(seed, seed^0x9e3779b97f4a7c15))} }

// sprand is a rows×cols matrix with the given density of N(0,1) entries.
func (g *gen) sprand(rows, cols int, density float64) *coo {
	m := newCOO(rows, cols)
	for j := 0; j < cols; j++ {
		for i := 0; i < rows; i++ {
			if g.r.Float64() < density {
				m.add(i, j, g.r.NormFloat64())
			}
		}
	}
	return m
}

func (g *gen) randn(n int) []float64 {
	v := make([]float64, n)
	for i := range v {
		v[i] = g.r.NormFloat64()
	}
	return v
}

// mulVec returns M x.
func (c *coo) mulVec(x []float64) []float64 {
	y := make([]float64, c.rows)
	for k := range c.v {
		y[c.i[k]] += c.v[k] * x[c.j[k]]
	}
	return y
}

// gram returns M Mᵀ + shift·I (the P of the random and equality classes).
func (c *coo) gram(shift float64) *coo {
	rows := make([][]float64, c.rows) // dense rows: the sums run in a fixed order
	for i := range rows {
		rows[i] = make([]float64, c.cols)
	}
	for k := range c.v {
		rows[c.i[k]][c.j[k]] += c.v[k]
	}
	p := newCOO(c.rows, c.rows)
	for a := 0; a < c.rows; a++ {
		for b := a; b < c.rows; b++ {
			s := 0.0
			for j, va := range rows[a] {
				s += va * rows[b][j]
			}
			if a == b {
				s += shift
			}
			if s != 0 {
				p.add(a, b, s)
				if a != b {
					p.add(b, a, s)
				}
			}
		}
	}
	return p
}

func inst(class, name string, p, a *coo, q, l, u []float64) Instance {
	return Instance{Name: name, Class: class, N: len(q), M: a.rows, P: p.csc(true), Q: q, A: a.csc(false), L: l, U: u}
}

func fill(n int, v float64) []float64 {
	o := make([]float64, n)
	for i := range o {
		o[i] = v
	}
	return o
}

// The classes follow the OSQP paper's section 8 as osqp_benchmarks implements
// them (problem_classes/*.py, opened for this): same structure, density 0.15,
// m = 10n, 100n data points; the random numbers are Go's, so the instances are
// not numpy's. Control uses QN = 10 Q, the alternative in the reference's own
// comments, instead of the Riccati solution.

func (g *gen) randomQP(n int) Instance {
	m := 10 * n
	P := g.sprand(n, n, 0.15).gram(0.01)
	q := g.randn(n)
	A := g.sprand(m, n, 0.15)
	v := g.randn(n)
	u := A.mulVec(v)
	for i := range u {
		u[i] += g.r.Float64()
	}
	return inst("random-qp", fmt.Sprintf("random-qp-n%d", n), P, A, q, fill(m, -infBound), u)
}

func (g *gen) eqQP(n int) Instance {
	m := n / 2
	P := g.sprand(n, n, 0.15).gram(0.01)
	q := g.randn(n)
	A := g.sprand(m, n, 0.15)
	b := A.mulVec(g.randn(n))
	return inst("eq-qp", fmt.Sprintf("eq-qp-n%d", n), P, A, q, b, append([]float64(nil), b...))
}

func (g *gen) portfolio(k int) Instance {
	n := 100 * k
	F := g.sprand(n, k, 0.5)
	d := make([]float64, n)
	for i := range d {
		d[i] = g.r.Float64() * math.Sqrt(float64(k))
	}
	mu := g.randn(n)
	N := n + k
	P := newCOO(N, N)
	for i := 0; i < n; i++ {
		P.add(i, i, 2*d[i])
	}
	for i := 0; i < k; i++ {
		P.add(n+i, n+i, 2)
	}
	q := make([]float64, N)
	for i := 0; i < n; i++ {
		q[i] = -mu[i]
	}
	A := newCOO(1+k+n, N)
	for i := 0; i < n; i++ {
		A.add(0, i, 1)
		A.add(1+k+i, i, 1)
	}
	for t := range F.v { // F' x - y = 0
		A.add(1+F.j[t], F.i[t], F.v[t])
	}
	for i := 0; i < k; i++ {
		A.add(1+i, n+i, -1)
	}
	l := append(append([]float64{1}, make([]float64, k)...), make([]float64, n)...)
	u := append(append([]float64{1}, make([]float64, k)...), fill(n, 1)...)
	return inst("portfolio", fmt.Sprintf("portfolio-k%d", k), P, A, q, l, u)
}

func (g *gen) lasso(n int) Instance {
	m := 100 * n
	Ad := g.sprand(m, n, 0.15)
	xt := make([]float64, n)
	for i := range xt {
		if g.r.Float64() > 0.5 {
			xt[i] = g.r.NormFloat64() / math.Sqrt(float64(n))
		}
	}
	bd := Ad.mulVec(xt)
	for i := range bd {
		bd[i] += g.r.NormFloat64()
	}
	lmax := 0.0
	for j := 0; j < n; j++ {
		s := 0.0
		for t := range Ad.v {
			if Ad.j[t] == j {
				s += Ad.v[t] * bd[Ad.i[t]]
			}
		}
		lmax = math.Max(lmax, math.Abs(s))
	}
	N := 2*n + m // (x, y, t)
	P := newCOO(N, N)
	for i := 0; i < m; i++ {
		P.add(n+i, n+i, 2)
	}
	q := make([]float64, N)
	for i := 0; i < n; i++ {
		q[n+m+i] = lmax / 5
	}
	A := newCOO(m+2*n, N)
	A.block(0, 0, Ad)
	A.block(0, n, eye(m).scaled(-1))
	A.block(m, 0, eye(n))
	A.block(m, n+m, eye(n).scaled(-1))
	A.block(m+n, 0, eye(n))
	A.block(m+n, n+m, eye(n))
	l := append(append(append([]float64{}, bd...), fill(n, -infBound)...), make([]float64, n)...)
	u := append(append(append([]float64{}, bd...), make([]float64, n)...), fill(n, infBound)...)
	return inst("lasso", fmt.Sprintf("lasso-n%d", n), P, A, q, l, u)
}

func (g *gen) huber(n int) Instance {
	m := 100 * n
	Ad := g.sprand(m, n, 0.15)
	xt := g.randn(n)
	for i := range xt {
		xt[i] /= math.Sqrt(float64(n))
	}
	bd := Ad.mulVec(xt)
	for i := range bd {
		if g.r.Float64() < 0.95 {
			bd[i] += 0.5 * g.r.NormFloat64()
		} else {
			bd[i] += 10 * g.r.Float64()
		}
	}
	N := n + 3*m // (x, z, r, s)
	P := newCOO(N, N)
	for i := 0; i < m; i++ {
		P.add(n+i, n+i, 1)
	}
	q := append(make([]float64, n+m), fill(2*m, 1)...)
	A := newCOO(3*m, N)
	A.block(0, 0, Ad)
	A.block(0, n, eye(m).scaled(-1))
	A.block(0, n+m, eye(m).scaled(-1))
	A.block(0, n+2*m, eye(m))
	A.block(m, n+m, eye(m))
	A.block(2*m, n+2*m, eye(m))
	l := append(append([]float64{}, bd...), make([]float64, 2*m)...)
	u := append(append([]float64{}, bd...), fill(2*m, infBound)...)
	return inst("huber", fmt.Sprintf("huber-n%d", n), P, A, q, l, u)
}

func (g *gen) svm(n int) Instance {
	m := 100 * n
	half := m / 2
	N := n + m
	P := newCOO(N, N)
	for i := 0; i < n; i++ {
		P.add(i, i, 1)
	}
	q := append(make([]float64, n), fill(m, 0.5)...)
	A := newCOO(2*m, N)
	for r := 0; r < m; r++ {
		sign, shift := 1.0, 1.0/float64(n)
		if r >= half {
			sign, shift = -1, -1/float64(n)
		}
		for j := 0; j < n; j++ {
			if g.r.Float64() < 0.15 { // diag(b) * (A/sqrt(n) + 1/n on the support)
				A.add(r, j, sign*(g.r.NormFloat64()/math.Sqrt(float64(n))+shift))
			}
		}
		A.add(r, n+r, -1)
		A.add(m+r, n+r, 1)
	}
	l := append(fill(m, -infBound), make([]float64, m)...)
	u := append(fill(m, -1), fill(m, infBound)...)
	return inst("svm", fmt.Sprintf("svm-n%d", n), P, A, q, l, u)
}

func (g *gen) control(nx int) Instance {
	nu, T := nx/2, 10
	if nu < 1 {
		nu = 1
	}
	// Dynamics: A = I + 0.1 N(0,1) scaled to spectral safety by shrinking the
	// off-diagonal part; B random dense. (The reference rescales the eigenvalues.)
	Ad := eye(nx)
	for i := 0; i < nx; i++ {
		for j := 0; j < nx; j++ {
			Ad.add(i, j, 0.1*g.r.NormFloat64()/math.Sqrt(float64(nx)))
		}
	}
	B := g.sprand(nx, nu, 1)
	Q := make([]float64, nx)
	for i := range Q {
		if g.r.Float64() < 0.7 {
			Q[i] = g.r.Float64()
		}
	}
	umax := make([]float64, nu)
	for i := range umax {
		umax[i] = g.r.Float64()
	}
	xmax := make([]float64, nx)
	for i := range xmax {
		xmax[i] = 1 + g.r.Float64()
	}
	x0 := make([]float64, nx)
	for i := range x0 {
		x0[i] = 0.5 * xmax[i] * (2*g.r.Float64() - 1)
	}
	nz := (T+1)*nx + T*nu
	P := newCOO(nz, nz)
	for t := 0; t <= T; t++ {
		for i := 0; i < nx; i++ {
			w := 2 * Q[i]
			if t == T {
				w = 20 * Q[i] // QN = 10 Q
			}
			P.add(t*nx+i, t*nx+i, w)
		}
	}
	for t := 0; t < T; t++ {
		for i := 0; i < nu; i++ {
			P.add((T+1)*nx+t*nu+i, (T+1)*nx+t*nu+i, 0.2)
		}
	}
	rows := (T+1)*nx + nz
	A := newCOO(rows, nz)
	for t := 0; t <= T; t++ {
		A.block(t*nx, t*nx, eye(nx).scaled(-1))
		if t > 0 {
			A.block(t*nx, (t-1)*nx, Ad)
			A.block(t*nx, (T+1)*nx+(t-1)*nu, B)
		}
	}
	A.block((T+1)*nx, 0, eye(nz))
	l := make([]float64, rows)
	u := make([]float64, rows)
	for i := 0; i < nx; i++ {
		l[i], u[i] = -x0[i], -x0[i]
	}
	for t := 0; t <= T; t++ {
		for i := 0; i < nx; i++ {
			l[(T+1)*nx+t*nx+i], u[(T+1)*nx+t*nx+i] = -xmax[i], xmax[i]
		}
	}
	for t := 0; t < T; t++ {
		for i := 0; i < nu; i++ {
			l[(T+1)*nx+(T+1)*nx+t*nu+i], u[(T+1)*nx+(T+1)*nx+t*nu+i] = -umax[i], umax[i]
		}
	}
	return inst("control", fmt.Sprintf("control-nx%d", nx), P, A, make([]float64, nz), l, u)
}

// denseQP is the dense positive definite class for the active set against the
// quadprog package: P = MMᵀ/n + I fully dense, 2n inequality rows and n/10
// equalities, feasible by construction.
func (g *gen) denseQP(n int) Instance {
	M := make([]float64, n*n)
	for i := range M {
		M[i] = g.r.NormFloat64()
	}
	P := newCOO(n, n) // MMᵀ/n + I, both triangles
	for a := 0; a < n; a++ {
		for b := 0; b <= a; b++ {
			s := 0.0
			for k := 0; k < n; k++ {
				s += M[a*n+k] * M[b*n+k]
			}
			s /= float64(n)
			if a == b {
				s++
			}
			P.add(a, b, s)
			if a != b {
				P.add(b, a, s)
			}
		}
	}
	q := g.randn(n)
	me, mi := n/10, 2*n
	A := g.sprand(me+mi, n, 1)
	x0 := g.randn(n)
	b := A.mulVec(x0)
	l := make([]float64, me+mi)
	u := make([]float64, me+mi)
	for i := 0; i < me; i++ {
		l[i], u[i] = b[i], b[i]
	}
	for i := me; i < me+mi; i++ {
		l[i], u[i] = -infBound, b[i]+g.r.Float64()
	}
	return inst("dense", fmt.Sprintf("dense-n%d", n), P, A, q, l, u)
}

// Sizes of the default set. Every instance stays below ~2·10⁵ non-zeros so that
// a round of the harness takes about a minute.
var (
	sparseSizes = map[string][]int{
		"random-qp": {20, 50, 100},
		"eq-qp":     {50, 100, 200},
		"portfolio": {5, 10, 20},
		"lasso":     {5, 10, 20},
		"huber":     {5, 10},
		"svm":       {5, 10, 20},
		"control":   {5, 10, 20},
	}
	denseSizes = []int{10, 50, 200, 500, 1000}
)

// Generate returns the instance set for the seed, in a fixed order.
func Generate(seed uint64, quick bool) []Instance {
	var out []Instance
	g := newGen(seed)
	dense := denseSizes
	if quick {
		dense = denseSizes[:2]
	}
	for _, c := range []string{"random-qp", "eq-qp", "portfolio", "lasso", "huber", "svm", "control"} {
		sizes := sparseSizes[c]
		if quick {
			sizes = sizes[:1]
		}
		for _, s := range sizes {
			switch c {
			case "random-qp":
				out = append(out, g.randomQP(s))
			case "eq-qp":
				out = append(out, g.eqQP(s))
			case "portfolio":
				out = append(out, g.portfolio(s))
			case "lasso":
				out = append(out, g.lasso(s))
			case "huber":
				out = append(out, g.huber(s))
			case "svm":
				out = append(out, g.svm(s))
			case "control":
				out = append(out, g.control(s))
			}
		}
	}
	for _, n := range dense {
		out = append(out, g.denseQP(n))
	}
	return out
}

// writeInstances writes one JSON file per instance into dir (created).
func writeInstances(dir string, set []Instance) error {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	for i := range set {
		b, err := json.Marshal(&set[i])
		if err != nil {
			return err
		}
		if err := os.WriteFile(filepath.Join(dir, set[i].Name+".json"), b, 0o644); err != nil {
			return err
		}
	}
	return nil
}
