package optimize

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strconv"
	"testing"
	"time"

	"github.com/go-sparse/sparse"
)

// The instances of tools/qpbench (generated into QPBENCH_DIR by `qpbench gen`):
// the Go side of the comparison with OSQP v1.0.0 and quadprog. Without the
// variable the benchmark is skipped. The harness runs it with -test.cpu 1.

type benchCSC struct {
	Rows, Cols int
	Indptr     []int
	Indices    []int
	Data       []float64
}

type benchInstance struct {
	Name  string
	Class string
	N, M  int
	P     benchCSC
	Q     []float64
	A     benchCSC
	L, U  []float64
}

func (c benchCSC) sparse(tb testing.TB, upper bool) *sparse.CSC[float64] {
	tb.Helper()
	ptr := []int{0}
	idx := []int32{}
	val := []float64{}
	for j := 0; j < c.Cols; j++ {
		for k := c.Indptr[j]; k < c.Indptr[j+1]; k++ {
			if !upper || c.Indices[k] <= j {
				idx = append(idx, int32(c.Indices[k]))
				val = append(val, c.Data[k])
			}
		}
		ptr = append(ptr, len(idx))
	}
	m, err := sparse.NewCSC(c.Rows, c.Cols, ptr, idx, val)
	if err != nil {
		tb.Fatal(err)
	}
	return m
}

// dense returns the row-major matrix; sym mirrors an upper triangle.
func (c benchCSC) dense(sym bool) []float64 {
	d := make([]float64, c.Rows*c.Cols)
	for j := 0; j < c.Cols; j++ {
		for k := c.Indptr[j]; k < c.Indptr[j+1]; k++ {
			d[c.Indices[k]*c.Cols+j] = c.Data[k]
			if sym {
				d[j*c.Cols+c.Indices[k]] = c.Data[k]
			}
		}
	}
	return d
}

func (in *benchInstance) sparseProblem(tb testing.TB) QPProblem {
	return QPProblem{PSparse: in.P.sparse(tb, true), Q: in.Q,
		Cons: QPConstraints{ASparse: in.A.sparse(tb, false), L: qpInf(in.L), U: qpInf(in.U)}}
}

func (in *benchInstance) denseProblem(tb testing.TB) QPProblem {
	return QPProblem{P: qpArr(tb, in.P.dense(true), in.N, in.N), Q: in.Q,
		Cons: QPConstraints{Linear: []LinearConstraint{{A: qpArr(tb, in.A.dense(false), in.M, in.N), Lb: qpInf(in.L), Ub: qpInf(in.U)}}}}
}

func loadBenchInstances(tb testing.TB, dir string) []benchInstance {
	tb.Helper()
	files, err := filepath.Glob(filepath.Join(dir, "*.json"))
	if err != nil || len(files) == 0 {
		tb.Fatalf("no instances in %s: %v", dir, err)
	}
	var out []benchInstance
	for _, f := range files {
		raw, err := os.ReadFile(f)
		if err != nil {
			tb.Fatal(err)
		}
		var in benchInstance
		if err := json.Unmarshal(raw, &in); err != nil {
			tb.Fatalf("%s: %v", f, err)
		}
		out = append(out, in)
	}
	return out
}

// benchAdmmMaxDense is the largest dense instance ADMM is timed on (the reference
// side stops there too).
const benchAdmmMaxDense = 1000

func BenchmarkQP(b *testing.B) {
	dir := os.Getenv("QPBENCH_DIR")
	if dir == "" {
		b.Skip("QPBENCH_DIR is not set (see tools/qpbench)")
	}
	eps := 1e-3
	if s := os.Getenv("QPBENCH_EPS"); s != "" {
		v, err := strconv.ParseFloat(s, 64)
		if err != nil {
			b.Fatal(err)
		}
		eps = v
	}
	for _, in := range loadBenchInstances(b, dir) {
		in := in
		if in.Class == "dense" {
			b.Run(in.Name+"/activeset", func(b *testing.B) {
				runBenchQP(b, in.denseProblem(b), &QPOptions{Method: QPActiveSet})
			})
			if in.N > benchAdmmMaxDense {
				continue
			}
		}
		b.Run(in.Name+"/admm", func(b *testing.B) {
			runBenchQP(b, in.sparseProblem(b), &QPOptions{Method: QPADMM, EpsAbs: eps, EpsRel: eps})
		})
	}
}

// runBenchQP times NewQPSolver (setup: scaling, KKT assembly, ordering, symbolic
// and first numeric factorisation for ADMM; the Cholesky factor of P for the active
// set) and Solve separately, and reports the time go-sparse spent in the LDLᵀ
// refactorisations and solves.
func runBenchQP(b *testing.B, pr QPProblem, opts *QPOptions) {
	var setup, solve time.Duration
	var res *QPResult
	qpTimeKKT, qpKKTAnalyseNS, qpKKTFactorNS, qpKKTSolveNS = true, 0, 0, 0
	defer func() { qpTimeKKT = false }()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		t0 := time.Now()
		s, err := NewQPSolver(pr, opts)
		t1 := time.Now()
		if err != nil {
			b.Fatal(err)
		}
		r, err := s.Solve()
		t2 := time.Now()
		if err != nil {
			b.Fatal(err)
		}
		setup += t1.Sub(t0)
		solve += t2.Sub(t1)
		res = r
	}
	n := float64(b.N)
	b.ReportMetric(float64(setup.Nanoseconds())/n, "setup-ns/op")
	b.ReportMetric(float64(solve.Nanoseconds())/n, "solve-ns/op")
	b.ReportMetric(float64(qpKKTAnalyseNS)/n, "ldl-analyse-ns/op")
	b.ReportMetric(float64(qpKKTFactorNS)/n, "ldl-factor-ns/op")
	b.ReportMetric(float64(qpKKTSolveNS)/n, "ldl-solve-ns/op")
	b.ReportMetric(float64(res.Nit), "iters")
	b.ReportMetric(res.Fun, "obj")
	b.ReportMetric(float64(res.Status), "status")
}

// The accounting of the LDLᵀ time sees the factorisations and solves of an ADMM
// run, and nothing when it is off.
func TestKKTTimingCounters(t *testing.T) {
	pr := QPProblem{P: qpArr(t, []float64{2, 0, 0, 2}, 2, 2), Q: []float64{-2, -5},
		Cons: QPConstraints{Linear: []LinearConstraint{{A: qpArr(t, []float64{1, 1}, 1, 2), Lb: []float64{-1e30}, Ub: []float64{1}}}}}
	run := func() {
		if _, err := Quadprog(pr, &QPOptions{Method: QPADMM}); err != nil {
			t.Fatal(err)
		}
	}
	qpKKTAnalyseNS, qpKKTFactorNS, qpKKTSolveNS = 0, 0, 0
	run()
	if qpKKTAnalyseNS != 0 || qpKKTFactorNS != 0 || qpKKTSolveNS != 0 {
		t.Errorf("counted with the accounting off: %d %d %d", qpKKTAnalyseNS, qpKKTFactorNS, qpKKTSolveNS)
	}
	qpTimeKKT = true
	defer func() { qpTimeKKT = false }()
	// A tiny problem can finish inside one tick of a coarse clock (Windows): run
	// until every counter has moved.
	for i := 0; i < 2000 && (qpKKTAnalyseNS <= 0 || qpKKTFactorNS <= 0 || qpKKTSolveNS <= 0); i++ {
		run()
	}
	if qpKKTAnalyseNS <= 0 || qpKKTFactorNS <= 0 || qpKKTSolveNS <= 0 {
		t.Errorf("analyse %d ns, factor %d ns, solve %d ns after a run", qpKKTAnalyseNS, qpKKTFactorNS, qpKKTSolveNS)
	}
}

// With the instance directory set, every instance of the set runs: this is the
// check the harness makes sure of before it times anything.
func TestBenchInstancesSolve(t *testing.T) {
	dir := os.Getenv("QPBENCH_DIR")
	if dir == "" {
		t.Skip("QPBENCH_DIR is not set")
	}
	for _, in := range loadBenchInstances(t, dir) {
		if in.N > benchAdmmMaxDense {
			continue
		}
		r, err := Quadprog(in.sparseProblem(t), &QPOptions{Method: QPADMM})
		if err != nil || r.Status != QPOptimal {
			t.Errorf("%s: %v, status %d", in.Name, err, statusOf(r))
		}
	}
}
