package optimize

import (
	"math"
	"testing"
)

// TestDEIntegralLimitsAreOpen pins the Nextafter of SciPy's
//
//	limits[0, integrality] = nextafter(ceil(lb) - 0.5, inf)
//	limits[1, integrality] = nextafter(floor(ub) + 0.5, -inf)
//
// (_differentialevolution.py, __init__, after the scale arguments are taken
// from the raw bounds). It has no black-box effect: the limits of an integral
// axis are read only by the degenerate-axis test (lo == hi, never true for
// the open interval and not for the closed one either) and are overwritten
// by the found value when polishing, so no fixture of SciPy's results can
// tell the two apart (removing the calls leaves all 296 + 82 runs
// unchanged). The interval is therefore checked directly: it is exactly the
// open one, strictly inside [lb - 0.5, ub + 0.5], and still holds a point
// that rounds to each of lb and ub.
func TestDEIntegralLimitsAreOpen(t *testing.T) {
	cases := [][2]float64{{1, 10}, {0, 7}, {-3, 4}, {2, 9}, {-8, -1}, {3, 3}, {-1, 1}, {0.2, 5.7}, {5, 6}, {1e15, 1e15 + 8}}
	for _, c := range cases {
		lb, ub := math.Ceil(c[0]), math.Floor(c[1])
		s, err := newDESolver(func(x []float64) float64 { return 0 },
			[]Bound{{c[0], c[1]}, {0, 1}}, &DEOptions{RNG: NewRNG(1), Integrality: []bool{true, false}})
		if err != nil {
			t.Fatalf("%v: %v", c, err)
		}
		wantLo, wantHi := math.Nextafter(lb-0.5, math.Inf(1)), math.Nextafter(ub+0.5, math.Inf(-1))
		if s.lo[0] != wantLo || s.hi[0] != wantHi {
			t.Errorf("%v: limits [%v, %v], want the open [%v, %v]", c, s.lo[0], s.hi[0], wantLo, wantHi)
		}
		if !(s.lo[0] > lb-0.5 && s.hi[0] < ub+0.5) {
			t.Errorf("%v: limits [%v, %v] are not strictly inside [%v, %v]", c, s.lo[0], s.hi[0], lb-0.5, ub+0.5)
		}
		if math.RoundToEven(s.lo[0]) != lb || math.RoundToEven(s.hi[0]) != ub {
			t.Errorf("%v: the end points round to %v and %v, want %v and %v", c, math.RoundToEven(s.lo[0]), math.RoundToEven(s.hi[0]), lb, ub)
		}
		// The closed end point lb - 0.5 rounds half to even, to lb - 1 when
		// lb is odd: the reason for the open interval.
		if math.Mod(lb, 2) != 0 && math.Abs(lb) < 1e15 && math.RoundToEven(lb-0.5) == lb {
			t.Errorf("%v: lb - 0.5 = %v rounds to lb itself; the case does not show the reason", c, lb-0.5)
		}
	}
}

// TestDEIntegralPopulationStaysInRange: with integer variables on tight
// bounds the objective is only ever evaluated at integers, for both seeding
// forms and updating modes.
func TestDEIntegralPopulationStaysInRange(t *testing.T) {
	for _, upd := range []DEUpdating{DEImmediate, DEDeferred} {
		for _, rng := range []*RandomSource{NewRNG(3), NewLegacyRNG(3)} {
			bad := 0
			r, err := DifferentialEvolution(func(x []float64) float64 {
				for _, v := range x {
					if v != math.Round(v) {
						bad++
					}
				}
				return x[0]*x[0] + x[1]*x[1]
			}, []Bound{{1, 3}, {1, 3}}, &DEOptions{RNG: rng, Integrality: []bool{true, true}, Updating: upd, MaxIter: Int(30), PopSize: 4})
			if err != nil {
				t.Fatal(err)
			}
			if bad != 0 || r.X[0] != 1 || r.X[1] != 1 {
				t.Errorf("updating %v: %d evaluations at non-integers, result %v", upd, bad, r.X)
			}
		}
	}
}
