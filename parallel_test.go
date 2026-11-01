package optimize

import (
	"errors"
	"math"
	"sync/atomic"
	"testing"
	"time"

	"github.com/go-ndarray/ndarray/random"
)

func TestEvalBatchDeterministic(t *testing.T) {
	pts := make([][]float64, 100)
	for i := range pts {
		pts[i] = []float64{float64(i), float64(i) * 0.5}
	}
	f := func(x []float64) float64 {
		if int(x[0])%7 == 0 { // uneven cost: scheduling differs between runs
			time.Sleep(50 * time.Microsecond)
		}
		return float64(x[0]*x[0]) + x[1]
	}
	ref := make([]float64, len(pts))
	evalBatch(f, pts, ref, 1)
	for _, w := range []int{0, 1, 2, 7, 64, 1000} {
		out := make([]float64, len(pts))
		evalBatch(f, pts, out, w)
		for i := range out {
			if math.Float64bits(out[i]) != math.Float64bits(ref[i]) {
				t.Fatalf("workers %d, point %d: %v != %v", w, i, out[i], ref[i])
			}
		}
	}
	evalBatch(f, nil, nil, 4) // empty batch
}

func TestParallelForPanicOnCaller(t *testing.T) {
	type boom struct{ i int }
	for _, w := range []int{1, 2, 8} {
		var calls atomic.Int64
		func() {
			defer func() {
				v := recover()
				b, ok := v.(boom)
				if !ok || b.i != 13 {
					t.Errorf("workers %d: recovered %v, want boom{13}", w, v)
				}
			}()
			parallelFor(1000, w, func(_, i int) {
				calls.Add(1)
				if i == 13 {
					panic(boom{13})
				}
			})
			t.Errorf("workers %d: no panic", w)
		}()
		if w > 1 && calls.Load() == 1000 {
			t.Errorf("workers %d: the batch was not drained early", w)
		}
	}
}

func TestParallelForLowestPanic(t *testing.T) {
	// every index panics: the lowest index seen wins, and with 1 worker that
	// is 0. With several workers the winner is the lowest among the started.
	defer func() {
		if v := recover(); v == nil {
			t.Error("no panic")
		}
	}()
	parallelFor(64, 4, func(_, i int) { panic(i) })
}

func TestParallelForWorkerScratch(t *testing.T) {
	// worker ids stay below effectiveWorkers.
	n, w := 50, 4
	var bad atomic.Bool
	parallelFor(n, w, func(k, _ int) {
		if k < 0 || k >= effectiveWorkers(n, w) {
			bad.Store(true)
		}
	})
	if bad.Load() {
		t.Error("worker id out of range")
	}
	if effectiveWorkers(0, 8) != 1 || effectiveWorkers(3, 8) != 3 || effectiveWorkers(9, -1) != 1 {
		t.Error("effectiveWorkers")
	}
}

func TestOptionPointers(t *testing.T) {
	if *Float(0) != 0 || *Float(2.5) != 2.5 || *Int(0) != 0 || *Int(7) != 7 {
		t.Error("pointer helpers")
	}
	a, b := Float(1), Float(1)
	if a == b {
		t.Error("Float must return distinct pointers")
	}
}

// the streams are NumPy's: first PCG64(SeedSequence(42)) double and
// RandomState(42) double.
func TestRandomSourceStreams(t *testing.T) {
	g := NewRNG(42)
	if want := random.DefaultRNG(42).Float64(); g.Float64() != want {
		t.Error("NewRNG is not Generator(PCG64(SeedSequence(42)))")
	}
	l := NewLegacyRNG(42)
	// numpy: np.random.RandomState(42).random_sample() = 0.3745401188473625
	if v := l.Float64(); v != 0.3745401188473625 {
		t.Errorf("NewLegacyRNG: %v", v)
	}
	for _, s := range []*RandomSource{NewRNG(1), NewLegacyRNG(1)} {
		u := s.Uniform(-2, 2)
		if u < -2 || u >= 2 {
			t.Error(u)
		}
		dst := make([]float64, 5)
		if err := s.FillUniform(dst, 0, 1); err != nil {
			t.Fatal(err)
		}
		if err := s.FillUniform(dst, 0, math.Inf(1)); !errors.Is(err, ErrInput) {
			t.Error("infinite range accepted", err)
		}
		if k := s.IntN(10); k < 0 || k >= 10 {
			t.Error(k)
		}
		ii := make([]int64, 8)
		if err := s.FillIntN(ii, 6); err != nil {
			t.Fatal(err)
		}
		if err := s.FillIntN(ii, 0); !errors.Is(err, ErrInput) {
			t.Error("high 0 accepted", err)
		}
		p := []int64{0, 1, 2, 3, 4, 5, 6, 7}
		s.ShuffleInt64(p)
		var sum int64
		for _, v := range p {
			sum += v
		}
		if sum != 28 {
			t.Error("shuffle lost elements", p)
		}
	}
	// uniform(low, high) scalar equals the one-element FillUniform
	a, b := NewRNG(9), NewRNG(9)
	one := make([]float64, 1)
	_ = b.FillUniform(one, 0.5, 1)
	if a.Uniform(0.5, 1) != one[0] {
		t.Error("Uniform differs from FillUniform")
	}
}

func TestRandomSourceAdapters(t *testing.T) {
	if RNGFromGenerator(random.DefaultRNG(3)).Float64() != random.DefaultRNG(3).Float64() {
		t.Error("RNGFromGenerator")
	}
	if RNGFromRandomState(random.NewRandomState(3)).Float64() != random.NewRandomState(3).Float64() {
		t.Error("RNGFromRandomState")
	}
	if v := RNGFromGenerator(nil).Float64(); v < 0 || v >= 1 {
		t.Error(v)
	}
	if v := RNGFromRandomState(nil).Float64(); v < 0 || v >= 1 {
		t.Error(v)
	}
	var nilSrc *RandomSource
	if v := nilSrc.orEntropy().Float64(); v < 0 || v >= 1 {
		t.Error(v)
	}
	if v := (&RandomSource{}).orEntropy().Float64(); v < 0 || v >= 1 {
		t.Error(v)
	}
	s := NewRNG(1)
	if s.orEntropy() != s {
		t.Error("orEntropy replaced a live source")
	}
}

func TestBruteWorkersIdentical(t *testing.T) {
	f := func(x []float64) float64 { return float64(x[0]*x[0]) + float64(x[1]*x[1]*x[1]*x[1]) - x[0]*x[1] }
	rs := []Range{{-2, 2, 0}, {-2, 2, 0}}
	ref, err := Brute(f, rs, &BruteOptions{Ns: 17, NoFinish: true})
	if err != nil {
		t.Fatal(err)
	}
	for _, w := range []int{2, 5, 64} {
		r, err := Brute(f, rs, &BruteOptions{Ns: 17, NoFinish: true, Workers: w})
		if err != nil {
			t.Fatal(err)
		}
		for i := range ref.Jout {
			if math.Float64bits(r.Jout[i]) != math.Float64bits(ref.Jout[i]) {
				t.Fatalf("workers %d, cell %d", w, i)
			}
		}
	}
}

func TestBrutePanicPropagates(t *testing.T) {
	for _, w := range []int{1, 4} {
		func() {
			defer func() {
				if v := recover(); v != "objective failed" {
					t.Errorf("workers %d: %v", w, v)
				}
			}()
			_, _ = Brute(func([]float64) float64 { panic("objective failed") },
				[]Range{{0, 1, 0}}, &BruteOptions{NoFinish: true, Workers: w})
		}()
	}
}

func TestBruteBounded(t *testing.T) {
	f := func([]float64) float64 { return 0 }
	big := math.MaxInt32
	cases := []struct {
		rs []Range
		o  *BruteOptions
	}{
		{[]Range{{0, 1, 0}}, &BruteOptions{Ns: big}},                           // Ns over the cap
		{[]Range{{0, 1, 0}, {0, 1, 0}, {0, 1, 0}}, &BruteOptions{Ns: 1 << 20}}, // product over the cap
		{[]Range{{0, 1e300, 1e-5}}, nil},                                       // step count over the cap
		{[]Range{{0, 1e300, 1e-5}, {0, 1, 1}}, &BruteOptions{NoFinish: true}},  // same, second axis
		{[]Range{{0, 1, 1e-300}}, nil},                                         // Inf count
	}
	for i, c := range cases {
		if _, err := Brute(f, c.rs, c.o); !errors.Is(err, ErrInput) {
			t.Errorf("case %d: %v", i, err)
		}
	}
}
