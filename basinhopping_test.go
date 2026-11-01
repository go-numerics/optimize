package optimize

import (
	"errors"
	"math"
	"testing"
)

func bhQuad(x, g []float64) float64 {
	g[0], g[1] = 2*x[0], 2*x[1]
	return float64(x[0]*x[0]) + float64(x[1]*x[1])
}

type fixedStep struct{ d float64 }

func (s fixedStep) TakeStep(x []float64) {
	for i := range x {
		x[i] += s.d
	}
}

type sizedStep struct {
	size float64
	n    int
}

func (s *sizedStep) TakeStep(x []float64)               { s.n++; x[0] += s.size }
func (s *sizedStep) StepSize() float64                  { return s.size }
func (s *sizedStep) SetStepSize(v float64)              { s.size = v }
func bhProblem() Problem                                { return Problem{FuncGrad: bhHimmelblau} }
func bhStart() []float64                                { return []float64{0, 0} }
func bhOpts(o BasinhoppingOptions) *BasinhoppingOptions { o.RNG = NewRNG(1); return &o }

func TestBasinhoppingFindsGlobal(t *testing.T) {
	r, err := Basinhopping(Problem{FuncGrad: bhCamel}, []float64{0.5, 0.5}, bhOpts(BasinhoppingOptions{NIter: Int(30)}))
	if err != nil {
		t.Fatal(err)
	}
	if !r.Success || math.Abs(r.Fun+1.0316284534898774) > 1e-6 || r.Nit != 30 || r.Message != bhMsgDone ||
		r.Lowest == nil || r.Nfev <= 0 {
		t.Errorf("%+v", r)
	}
}

func TestBasinhoppingErrors(t *testing.T) {
	good := bhOpts(BasinhoppingOptions{NIter: Int(1)})
	_ = good
	cases := map[string]*BasinhoppingOptions{
		"target":    {TargetAcceptRate: 1.5},
		"factor":    {StepwiseFactor: -0.1},
		"niter":     {NIter: Int(-1)},
		"interval":  {Interval: -1},
		"nsuccess":  {NIterSuccess: -1},
		"stepsize":  {StepSize: -1},
		"stepNaN":   {StepSize: math.NaN()},
		"both":      {AcceptTest: func(float64, []float64, float64, []float64) Acceptance { return Accept }, AcceptTestResult: func(a, b *Result) Acceptance { return Accept }},
		"method":    {Method: "nope"},
		"bad bound": {MinimizeOptions: &Options{Bounds: []Bound{{0, 1}}}},
	}
	for name, o := range cases {
		o.RNG = NewRNG(1)
		if _, err := Basinhopping(bhProblem(), bhStart(), o); err == nil {
			t.Errorf("%s: accepted", name)
		}
	}
	if _, err := Basinhopping(Problem{}, bhStart(), nil); !errors.Is(err, ErrInput) {
		t.Error(err)
	}
	// an error in a later local minimisation is returned too
	n := 0
	p := Problem{FuncGrad: func(x, g []float64) float64 { return bhQuad(x, g) }}
	step := stepFunc(func(x []float64) { n++; x[0] = math.NaN() })
	// the second local minimisation fails: the error is returned, no result
	calls := 0
	boom := errors.New("boom")
	bhMinimize = func(p Problem, x []float64, m Method, o *Options) (*Result, error) {
		if calls++; calls == 2 {
			return nil, boom
		}
		return Minimize(p, x, m, o)
	}
	defer func() { bhMinimize = Minimize }()
	r, err := Basinhopping(p, bhStart(), bhOpts(BasinhoppingOptions{NIter: Int(2), TakeStep: step}))
	if !errors.Is(err, boom) || r != nil {
		t.Errorf("a failing local minimisation must be returned: %v %v", err, r)
	}
}

type stepFunc func(x []float64)

func (f stepFunc) TakeStep(x []float64) { f(x) }

func TestBasinhoppingStepsAndTests(t *testing.T) {
	// a plain StepTaker is used as is; a StepSizer is adapted
	plain := 0
	r, err := Basinhopping(bhProblem(), bhStart(), bhOpts(BasinhoppingOptions{NIter: Int(5), TakeStep: stepFunc(func(x []float64) { plain++; x[0] += 0.1 })}))
	if err != nil || plain != 5 || r.Nit != 5 {
		t.Errorf("%v %d %+v", err, plain, r)
	}
	st := &sizedStep{size: 1}
	if _, err := Basinhopping(bhProblem(), bhStart(), bhOpts(BasinhoppingOptions{NIter: Int(6), Interval: 2, TakeStep: st})); err != nil {
		t.Fatal(err)
	}
	if st.n != 6 || st.size == 1 {
		t.Errorf("the step size was not adapted: %+v", st)
	}
	// accept tests: ForceAccept skips Metropolis (no draw), Reject still draws
	for _, o := range []BasinhoppingOptions{
		{AcceptTest: func(float64, []float64, float64, []float64) Acceptance { return ForceAccept }},
		{AcceptTest: func(float64, []float64, float64, []float64) Acceptance { return Reject }},
		{AcceptTestResult: func(n, o *Result) Acceptance { return Accept }},
	} {
		o.NIter = Int(4)
		if _, err := Basinhopping(bhProblem(), bhStart(), bhOpts(o)); err != nil {
			t.Error(err)
		}
	}
	// a ForceAccept run takes every step: monotonic T=0 would reject some
	acc := 0
	_, _ = Basinhopping(bhProblem(), bhStart(), bhOpts(BasinhoppingOptions{NIter: Int(8), T: Float(0), StepSize: 3,
		AcceptTest: func(float64, []float64, float64, []float64) Acceptance { return ForceAccept },
		Callback: func(_ []float64, _ float64, a bool) bool {
			if a {
				acc++
			}
			return false
		}}))
	if acc != 9 { // the initial call plus 8 accepted steps
		t.Errorf("%d accepted", acc)
	}
}

func TestBasinhoppingStops(t *testing.T) {
	calls := 0
	r, _ := Basinhopping(bhProblem(), bhStart(), bhOpts(BasinhoppingOptions{NIter: Int(50),
		Callback: func(x []float64, f float64, a bool) bool { calls++; return calls == 4 }}))
	// the initial call's return value is ignored; the 4th call is step 3
	if r.Message != bhMsgCallback || r.Nit != 3 {
		t.Errorf("%+v", r)
	}
	r, _ = Basinhopping(bhProblem(), bhStart(), bhOpts(BasinhoppingOptions{NIter: Int(50), NIterSuccess: 3}))
	if r.Message != bhMsgSuccess || r.Nit >= 50 {
		t.Errorf("%+v", r)
	}
	r, _ = Basinhopping(bhProblem(), bhStart(), bhOpts(BasinhoppingOptions{NIter: Int(0)}))
	if r.Nit != 1 { // Python's loop variable: nit = 1 even with niter = 0
		t.Errorf("%+v", r)
	}
	r, _ = Basinhopping(bhProblem(), bhStart(), bhOpts(BasinhoppingOptions{NIter: Int(math.MaxInt - 1), NIterSuccess: 2}))
	if r.Message != bhMsgSuccess {
		t.Errorf("%+v", r)
	}
	r, _ = Basinhopping(bhProblem(), bhStart(), bhOpts(BasinhoppingOptions{NIter: Int(math.MaxInt - 1), Callback: func([]float64, float64, bool) bool { return true }}))
	if r.Message != bhMsgCallback {
		t.Errorf("%+v", r)
	}
}

func TestBasinhoppingDefaultMethod(t *testing.T) {
	// bounds select L-BFGS-B, constraints SLSQP, nothing BFGS
	for _, mo := range []*Options{
		nil,
		{Bounds: []Bound{{-5, 5}, {-5, 5}}},
		{Constraints: []ConstraintSpec{Constraint{Kind: Ineq, M: 1, Func: func(x, c []float64) { c[0] = x[0] + x[1] + 1 }}}},
	} {
		r, err := Basinhopping(Problem{FuncGrad: bhQuad}, []float64{1, 1}, bhOpts(BasinhoppingOptions{NIter: Int(2), MinimizeOptions: mo}))
		if err != nil || r.Lowest == nil {
			t.Errorf("%v %+v", err, mo)
		}
	}
}

func TestBasinhoppingFailureCounted(t *testing.T) {
	// a one-iteration minimiser cannot converge: every minimisation fails
	r, err := Basinhopping(bhProblem(), []float64{3, 3}, bhOpts(BasinhoppingOptions{NIter: Int(3),
		MinimizeOptions: &Options{MaxIter: 1}}))
	if err != nil || r.MinimizationFailures != 4 || r.Success {
		t.Errorf("%v %+v", err, r)
	}
	// a failed incumbent lets a failed step through, and a later success replaces the stored minimum
}

func TestBasinhoppingEntropyRNG(t *testing.T) {
	if _, err := Basinhopping(bhProblem(), bhStart(), &BasinhoppingOptions{NIter: Int(2)}); err != nil {
		t.Error(err)
	}
}
