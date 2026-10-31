package optimize

import (
	"encoding/binary"
	"math"
	"testing"
)

// FuzzSLSQP builds a small problem from raw bytes — a quadratic objective,
// bounds and linear and nonlinear constraints whose coefficients may be NaN,
// ±Inf, huge, zero or contradictory — and checks that Minimize neither
// panics nor exceeds its evaluation budget.
func FuzzSLSQP(f *testing.F) {
	f.Add([]byte{2, 1, 1, 0}, []byte("\x00\x00\x00\x00\x00\x00\xf0\x3f"))
	f.Add([]byte{3, 2, 0, 1}, []byte("\x00\x00\x00\x00\x00\x00\xf8\x7f\x00\x00\x00\x00\x00\x00\xf0\x7f"))
	f.Add([]byte{2, 1, 1, 8 | 1}, []byte("\x00\x00\x00\x00\x00\x00\xf0\x3f"))
	f.Add([]byte{3, 0, 2, 8 | 2}, []byte("\x00\x00\x00\x00\x00\x00\xf8\x7f\x00\x00\x00\x00\x00\x00\xf0\x7f"))
	f.Add([]byte{1, 3, 2, 3}, []byte("\xff\xff\xff\xff\xff\xff\xef\x7f\x01\x00\x00\x00\x00\x00\x00\x00"))
	f.Fuzz(func(t *testing.T, shape, raw []byte) {
		if len(shape) < 4 {
			return
		}
		n := 1 + int(shape[0])%4
		meq := int(shape[1]) % 3
		mineq := int(shape[2]) % 3
		flags := shape[3]
		pos := 0
		next := func() float64 {
			if pos+8 > len(raw) {
				pos++
				return float64(pos%7) - 3
			}
			v := math.Float64frombits(binary.LittleEndian.Uint64(raw[pos:]))
			pos += 8
			return v
		}
		center := make([]float64, n)
		weight := make([]float64, n)
		x0 := make([]float64, n)
		for i := range n {
			center[i], weight[i], x0[i] = next(), next(), next()
		}
		obj := func(x []float64) float64 {
			s := 0.0
			for i, v := range x {
				d := v - center[i]
				s += float64(weight[i] * d * d)
			}
			return s
		}
		var cons []ConstraintSpec
		for k := 0; k < meq+mineq; k++ {
			row := make([]float64, n)
			for i := range row {
				row[i] = next()
			}
			rhs, curv := next(), next()
			kind := Ineq
			if k < meq {
				kind = Eq
			}
			c := Constraint{Kind: kind, M: 1, Func: func(x, c []float64) {
				c[0] = dot(row, x) - rhs + float64(curv*dot(x, x))
			}}
			if flags&1 != 0 {
				c.Jac = func(x, j []float64) {
					for i := range j {
						j[i] = row[i] + float64(2*curv*x[i])
					}
				}
			}
			cons = append(cons, c)
		}
		o := &Options{Constraints: cons, MaxIter: 30}
		if flags&2 != 0 {
			o.Bounds = make([]Bound, n)
			for i := range o.Bounds {
				o.Bounds[i] = Bound{next(), next()}
			}
		}
		p := Problem{Func: obj}
		if flags&4 != 0 {
			p.Grad = func(x, g []float64) {
				for i, v := range x {
					g[i] = float64(2 * weight[i] * (v - center[i]))
				}
			}
		}
		evals := 0
		inner := p.Func
		p.Func = func(x []float64) float64 { evals++; return inner(x) }
		method := SLSQP
		// 30 iterations, at most 11 line-search points each, and n+1
		// evaluations per finite-difference gradient.
		limit := 1 + 30*(11+n+1)
		if flags&8 != 0 {
			// trust-constr: at most a trial point and a second-order
			// correction per iteration, each with an FD gradient (BFGS
			// updates happen at every new point), plus the start.
			method = TrustConstr
			limit = 1 + n + 30*2*(n+1)
		}
		r, err := Minimize(p, x0, method, o)
		if err != nil {
			return
		}
		if evals > limit {
			t.Fatalf("%d evaluations > %d", evals, limit)
		}
		// All variables fixed by bounds: no multipliers, as in SciPy.
		if len(r.X) != n || (len(r.Multipliers) != meq+mineq && r.Nit != 0) {
			t.Fatalf("result shape %d %d", len(r.X), len(r.Multipliers))
		}
	})
}
