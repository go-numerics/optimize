package optimize

import (
	"encoding/binary"
	"errors"
	"math"
	"testing"

	"github.com/go-sparse/sparse"
)

// qpBytes hands out float64s from fuzz input. In raw mode the bits are taken
// as they come (NaN, ±Inf, subnormals, 1e300 all occur); otherwise they are
// small integers, which keeps the generated problems solvable so that the
// success path is exercised too.
type qpBytes struct {
	b   []byte
	raw bool
}

func (r *qpBytes) next() float64 {
	if len(r.b) < 8 {
		r.b = append(r.b, make([]byte, 8-len(r.b))...)
	}
	u := binary.LittleEndian.Uint64(r.b[:8])
	r.b = r.b[8:]
	if r.raw {
		return math.Float64frombits(u)
	}
	return float64(int16(u)) / 64
}

func isQPError(err error) bool {
	for _, e := range []error{ErrDimension, ErrInput, ErrBounds, ErrMethod} {
		if errors.Is(err, e) {
			return true
		}
	}
	return false
}

func checkQPOutcome(t *testing.T, n int, r *QPResult, err error) {
	t.Helper()
	if err != nil {
		if r != nil || !isQPError(err) {
			t.Fatalf("error %v with result %v", err, r)
		}
		return
	}
	if r == nil || len(r.X) != n {
		t.Fatalf("result %+v for n = %d", r, n)
	}
	if r.Success != (r.Status == QPOptimal) || r.Nit < 0 {
		t.Fatalf("inconsistent result %+v", r)
	}
	if r.Success {
		for _, v := range r.X {
			if math.IsNaN(v) || math.IsInf(v, 0) {
				t.Fatalf("success with x = %v (%+v)", r.X, r)
			}
		}
		if !(r.Optimality <= math.MaxFloat64) || !(r.ConstrViolation <= math.MaxFloat64) {
			t.Fatalf("success with optimality %g, violation %g", r.Optimality, r.ConstrViolation)
		}
	}
}

// FuzzQuadprogDense builds small dense problems (n <= 8, m <= 12) from bytes,
// sometimes with a singular or indefinite P and injected NaN, ±Inf and huge
// values, and checks that Quadprog never panics and returns an error of the
// documented kinds or a consistent result.
func FuzzQuadprogDense(f *testing.F) {
	f.Add([]byte("\x01\x00\x02\x00\x03\x00\x04\x00\x05\x00\x06\x00\x07\x00\x08\x00"), uint8(2), uint8(1), uint8(0))
	f.Add([]byte{}, uint8(0), uint8(0), uint8(0))
	f.Add([]byte("0123456789abcdef0123456789abcdef"), uint8(7), uint8(12), uint8(1))
	f.Add([]byte("\xff\xff\xff\xff\xff\xff\xff\x7f\x00\x00\x00\x00\x00\x00\xf0\x7f"), uint8(3), uint8(2), uint8(1|2|4))
	f.Add([]byte("zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz"), uint8(4), uint8(3), uint8(2|4|8))
	f.Fuzz(func(t *testing.T, data []byte, n8, m8, flags uint8) {
		n, m := int(n8)%8+1, int(m8)%13
		rd := &qpBytes{b: data, raw: flags&1 != 0}
		vec := func(k int) []float64 {
			v := make([]float64, k)
			for i := range v {
				v[i] = rd.next()
			}
			return v
		}
		p := QPProblem{Q: vec(n)}
		l := vec(n * n)
		pm := make([]float64, n*n)
		if flags&2 != 0 { // raw P: singular, indefinite, asymmetric or non-finite
			copy(pm, l)
		} else { // P = L·Lᵀ + I
			for i := 0; i < n; i++ {
				for j := 0; j < n; j++ {
					s := 0.0
					for k := 0; k < n; k++ {
						s += float64(l[i*n+k] * l[j*n+k])
					}
					pm[i*n+j] = s
				}
				pm[i*n+i]++
			}
		}
		p.P = qpArr(t, pm, n, n)
		if m > 0 {
			p.Cons.AEq, p.Cons.BEq = qpArr(t, vec(m*n), m, n), qpArr(t, vec(m), m)
		}
		if flags&4 != 0 {
			p.Cons.AUb, p.Cons.BUb = qpArr(t, vec(n), n), qpArr(t, vec(1), 1)
		}
		if flags&8 != 0 {
			b := make([]Bound, n)
			for i := range b {
				lo, hi := rd.next(), rd.next()
				b[i] = Bound{lo, hi}
			}
			p.Cons.Bounds = b
		}
		o := &QPOptions{}
		if flags&16 != 0 {
			o.Method = QPActiveSet
		}
		r, err := Quadprog(p, o)
		checkQPOutcome(t, n, r, err)
	})
}

// FuzzQuadprogSparse feeds malformed CSC matrices (built as struct literals,
// bypassing NewCSC) as PSparse and ASparse: unsorted or duplicate indices,
// indices out of range, a wrong Indptr, lower-triangle entries, NaN values.
// Every one must be an error, never a panic.
func FuzzQuadprogSparse(f *testing.F) {
	f.Add([]byte{0, 1, 2, 0, 1, 0, 1, 0}, uint8(2), uint8(2), uint8(1))
	f.Add([]byte{0, 0, 0, 0}, uint8(1), uint8(1), uint8(0))
	f.Add([]byte{0, 2, 4, 1, 0, 1, 0, 9, 9, 9}, uint8(2), uint8(2), uint8(3))
	f.Add([]byte{255, 255, 255, 255, 255}, uint8(3), uint8(3), uint8(2))
	f.Fuzz(func(t *testing.T, data []byte, n8, rows8, flags uint8) {
		n, rows := int(n8)%5+1, int(rows8)%5
		at := func(i int) int {
			if len(data) == 0 {
				return 0
			}
			return int(int8(data[i%len(data)]))
		}
		mk := func(r, c, off int) *sparse.CSC[float64] {
			ptr := make([]int, c+1)
			for j := range ptr {
				ptr[j] = at(off + j) // not sorted, not starting at 0: malformed on purpose
				if flags&1 != 0 {
					ptr[j] = j * 2
				}
			}
			k := 2 * c
			idx, val := make([]int32, k), make([]float64, k)
			for i := range idx {
				idx[i] = int32(at(off + c + i))
				val[i] = float64(at(off+c+k+i)) / 8
				if flags&2 != 0 && i == 0 {
					val[i] = math.NaN()
				}
			}
			return &sparse.CSC[float64]{Rows: r, Cols: c, Indptr: ptr, Indices: idx, Data: val}
		}
		p := QPProblem{Q: make([]float64, n), PSparse: mk(n, n, 0)}
		if rows > 0 {
			p.Cons.ASparse = mk(rows, n, 3)
			p.Cons.U = make([]float64, rows)
		}
		r, err := Quadprog(p, &QPOptions{Method: QPMethod([]string{"", "admm", "active-set"}[int(flags)%3])})
		checkQPOutcome(t, n, r, err)
	})
}
