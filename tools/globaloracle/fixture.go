package main

import (
	"encoding/json"
	"fmt"
	"io"
	"math"
	"sort"
	"strconv"
	"strings"
)

// Hex is a float64 stored in JSON as its IEEE-754 bit pattern, "0x" followed
// by 16 hexadecimal digits, so that NaN, ±Inf and signed zeros survive and
// comparisons are exact.
type Hex float64

// MarshalJSON writes the bit pattern.
func (h Hex) MarshalJSON() ([]byte, error) {
	return []byte(fmt.Sprintf(`"0x%016x"`, math.Float64bits(float64(h)))), nil
}

// UnmarshalJSON reads a bit pattern written by MarshalJSON; anything else is
// an error.
func (h *Hex) UnmarshalJSON(b []byte) error {
	var s string
	if err := json.Unmarshal(b, &s); err != nil {
		return fmt.Errorf("hex float: %w", err)
	}
	if len(s) != 18 || !strings.HasPrefix(s, "0x") {
		return fmt.Errorf("hex float %q: want 0x and 16 digits", s)
	}
	u, err := strconv.ParseUint(s[2:], 16, 64)
	if err != nil {
		return fmt.Errorf("hex float %q: %w", s, err)
	}
	*h = Hex(math.Float64frombits(u))
	return nil
}

// Fixture is the content of testdata/global/<method>.json: SciPy's results
// of one global method on the problems of global.py and the environment they
// were produced in.
type Fixture struct {
	Method  string `json:"method"`
	Scipy   string `json:"scipy"`
	Numpy   string `json:"numpy"`
	Python  string `json:"python"`
	Machine string `json:"machine"`
	System  string `json:"system"`
	Cases   []Case `json:"cases"`
}

// Options are the keyword arguments of scipy.optimize.direct, defaults
// resolved (maxfun is 1000*N when SciPy was given None). FMin is nil for
// SciPy's default of -inf.
type Options struct {
	Eps      float64  `json:"eps"`
	Maxfun   int      `json:"maxfun"`
	Maxiter  int      `json:"maxiter"`
	FMin     *float64 `json:"f_min"`
	FMinRtol float64  `json:"f_min_rtol"`
	VolTol   float64  `json:"vol_tol"`
	LenTol   float64  `json:"len_tol"`
}

// Case is one run of scipy.optimize.direct.
type Case struct {
	Name    string `json:"name"`
	Problem string `json:"problem"`
	// Family is "arith" (products and sums only: bit-exact comparison) or
	// "trans" (calls libm: comparison with a tolerance).
	Family   string    `json:"family"`
	Dim      int       `json:"dim"`
	Lower    []float64 `json:"lower"`
	Upper    []float64 `json:"upper"`
	Locally  bool      `json:"locally_biased"`
	Options  Options   `json:"options"`
	Error    *string   `json:"error"`
	X        []Hex     `json:"x,omitempty"`
	Fun      *Hex      `json:"fun,omitempty"`
	Nfev     int       `json:"nfev"`
	Nit      int       `json:"nit"`
	Status   int       `json:"status"`
	Success  bool      `json:"success"`
	Message  string    `json:"message"`
	Trace    [][]Hex   `json:"trace"`
	Ncallbck int       `json:"ncallbacks"`
}

// Decode reads a fixture, rejecting unknown fields, a fixture without
// cases, duplicate or missing case names, and cases whose arrays disagree
// with their dimension.
func Decode(r io.Reader) (*Fixture, error) {
	dec := json.NewDecoder(r)
	dec.DisallowUnknownFields()
	var f Fixture
	if err := dec.Decode(&f); err != nil {
		return nil, fmt.Errorf("fixture: %w", err)
	}
	if len(f.Cases) == 0 {
		return nil, fmt.Errorf("fixture: no cases")
	}
	seen := map[string]bool{}
	for _, c := range f.Cases {
		if c.Name == "" || seen[c.Name] {
			return nil, fmt.Errorf("fixture: missing or duplicate case name %q", c.Name)
		}
		seen[c.Name] = true
		if c.Dim < 1 || len(c.Lower) != c.Dim || len(c.Upper) != c.Dim {
			return nil, fmt.Errorf("fixture: case %q: bounds do not match dimension %d", c.Name, c.Dim)
		}
		if c.Error == nil {
			if len(c.X) != c.Dim || c.Fun == nil {
				return nil, fmt.Errorf("fixture: case %q: x or fun missing", c.Name)
			}
		}
		for _, t := range c.Trace {
			if len(t) != c.Dim {
				return nil, fmt.Errorf("fixture: case %q: trace entry of length %d, dimension %d", c.Name, len(t), c.Dim)
			}
		}
	}
	return &f, nil
}

// Encode writes f as indented JSON with a final newline, the committed form.
func Encode(w io.Writer, f *Fixture) error {
	enc := json.NewEncoder(w)
	enc.SetEscapeHTML(false)
	enc.SetIndent("", " ")
	return enc.Encode(f)
}

// Disagreement is one difference between two fixtures.
type Disagreement struct {
	Case, What string
}

func (d Disagreement) String() string { return d.Case + ": " + d.What }

func sameBits(a, b float64) bool { return math.Float64bits(a) == math.Float64bits(b) }

// Diff compares two fixtures case by case, exactly: the status, success,
// message, nfev, nit, the bit patterns of x and fun, the options, and the
// recorded trace. Cases present on one side only are reported too. The
// result is sorted by case name.
func Diff(a, b *Fixture) []Disagreement {
	byName := map[string]Case{}
	for _, c := range b.Cases {
		byName[c.Name] = c
	}
	var out []Disagreement
	add := func(name, format string, args ...any) {
		out = append(out, Disagreement{name, fmt.Sprintf(format, args...)})
	}
	inA := map[string]bool{}
	for _, ca := range a.Cases {
		inA[ca.Name] = true
		cb, ok := byName[ca.Name]
		if !ok {
			add(ca.Name, "only in the first fixture")
			continue
		}
		if (ca.Error == nil) != (cb.Error == nil) || (ca.Error != nil && *ca.Error != *cb.Error) {
			add(ca.Name, "error differs")
			continue
		}
		if ca.Error != nil {
			continue
		}
		if !optionsEqual(ca.Options, cb.Options) {
			add(ca.Name, "options differ: %+v, other %+v", ca.Options, cb.Options)
		}
		if ca.Status != cb.Status || ca.Success != cb.Success {
			add(ca.Name, "status %d success %v, other %d %v", ca.Status, ca.Success, cb.Status, cb.Success)
		}
		if ca.Message != cb.Message {
			add(ca.Name, "message %q, other %q", ca.Message, cb.Message)
		}
		if ca.Nit != cb.Nit || ca.Nfev != cb.Nfev {
			add(ca.Name, "nit/nfev %d/%d, other %d/%d", ca.Nit, ca.Nfev, cb.Nit, cb.Nfev)
		}
		if !sameBits(float64(*ca.Fun), float64(*cb.Fun)) {
			add(ca.Name, "fun %.17g, other %.17g", float64(*ca.Fun), float64(*cb.Fun))
		}
		if len(ca.X) != len(cb.X) {
			add(ca.Name, "len(x) %d, other %d", len(ca.X), len(cb.X))
		} else {
			for i := range ca.X {
				if !sameBits(float64(ca.X[i]), float64(cb.X[i])) {
					add(ca.Name, "x[%d] %.17g, other %.17g", i, float64(ca.X[i]), float64(cb.X[i]))
					break
				}
			}
		}
		if i, ok := traceDiffers(ca.Trace, cb.Trace); ok {
			add(ca.Name, "trace differs at iteration %d", i)
		}
	}
	for _, cb := range b.Cases {
		if !inA[cb.Name] {
			add(cb.Name, "only in the second fixture")
		}
	}
	sort.SliceStable(out, func(i, j int) bool { return out[i].Case < out[j].Case })
	return out
}

// optionsEqual compares options bit for bit (they are finite, but a
// fixture could hold anything).
func optionsEqual(a, b Options) bool {
	if (a.FMin == nil) != (b.FMin == nil) || (a.FMin != nil && !sameBits(*a.FMin, *b.FMin)) {
		return false
	}
	return a.Maxfun == b.Maxfun && a.Maxiter == b.Maxiter &&
		sameBits(a.Eps, b.Eps) && sameBits(a.FMinRtol, b.FMinRtol) &&
		sameBits(a.VolTol, b.VolTol) && sameBits(a.LenTol, b.LenTol)
}

// traceDiffers reports the first iteration at which two traces differ in
// length or in a bit.
func traceDiffers(a, b [][]Hex) (int, bool) {
	for i := 0; i < len(a) || i < len(b); i++ {
		if i >= len(a) || i >= len(b) || len(a[i]) != len(b[i]) {
			return i, true
		}
		for j := range a[i] {
			if !sameBits(float64(a[i][j]), float64(b[i][j])) {
				return i, true
			}
		}
	}
	return 0, false
}
