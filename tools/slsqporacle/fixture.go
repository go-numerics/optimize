package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"sort"
)

// Fixture is the content of testdata/slsqp.json or testdata/trustconstr.json:
// SciPy's results with one method on the problems of hs.py and the
// environment they were produced in.
type Fixture struct {
	Method  string `json:"method,omitempty"`
	Scipy   string `json:"scipy"`
	Numpy   string `json:"numpy"`
	Python  string `json:"python"`
	Machine string `json:"machine"`
	System  string `json:"system"`
	Cases   []Case `json:"cases"`
}

// Case is one minimize run.
type Case struct {
	Name        string             `json:"name"`
	Problem     string             `json:"problem"`
	Jac         bool               `json:"jac"`
	Options     map[string]float64 `json:"options"`
	Tol         *float64           `json:"tol"`
	X           []float64          `json:"x"`
	Fun         float64            `json:"fun"`
	Nit         int                `json:"nit"`
	Nfev        int                `json:"nfev"`
	Njev        int                `json:"njev"`
	Status      int                `json:"status"`
	Success     bool               `json:"success"`
	Message     string             `json:"message"`
	Multipliers []float64          `json:"multipliers"`
	Maxcv       float64            `json:"maxcv"`
	Fstar       *float64           `json:"fstar"`
	// Error is SciPy's exception when minimize raised instead of returning.
	Error string `json:"error,omitempty"`
	// trust-constr only.
	Optimality      *float64 `json:"optimality,omitempty"`
	ConstrViolation *float64 `json:"constr_violation,omitempty"`
	CgNiter         *int     `json:"cg_niter,omitempty"`
	Nhev            *int     `json:"nhev,omitempty"`
}

// Decode reads a fixture, rejecting unknown fields, a fixture without cases
// and duplicate case names.
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
	}
	return &f, nil
}

// Encode writes f as indented JSON with a final newline, the committed form.
func Encode(w io.Writer, f *Fixture) error {
	b, err := json.Marshal(f)
	if err != nil {
		return err
	}
	var out bytes.Buffer
	if err := json.Indent(&out, b, "", " "); err != nil {
		return err
	}
	out.WriteByte('\n')
	_, err = w.Write(out.Bytes())
	return err
}

// Disagreement is one difference between two fixtures.
type Disagreement struct {
	Case, What string
}

func (d Disagreement) String() string { return d.Case + ": " + d.What }

// Tolerances of Diff: x and multipliers relative to 1+|value|, f relative.
const (
	xTol   = 1e-7
	funTol = 1e-9
)

func near(a, b, tol float64) bool {
	if math.IsNaN(a) || math.IsNaN(b) {
		return math.IsNaN(a) && math.IsNaN(b)
	}
	return math.Abs(a-b) <= tol*(1+math.Abs(b))
}

// Diff compares two fixtures case by case: cases missing on either side, and
// for each common case the status, success, path (nit/nfev/njev), objective,
// solution and multipliers. The result is sorted by case name.
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
		if ca.Error != cb.Error {
			add(ca.Name, "error %q, other %q", ca.Error, cb.Error)
			continue
		}
		if ca.Status != cb.Status || ca.Success != cb.Success {
			add(ca.Name, "status %d success %v, other %d %v", ca.Status, ca.Success, cb.Status, cb.Success)
		}
		if ca.Nit != cb.Nit || ca.Nfev != cb.Nfev || ca.Njev != cb.Njev {
			add(ca.Name, "nit/nfev/njev %d/%d/%d, other %d/%d/%d", ca.Nit, ca.Nfev, ca.Njev, cb.Nit, cb.Nfev, cb.Njev)
		}
		if !near(ca.Fun, cb.Fun, funTol) {
			add(ca.Name, "fun %.17g, other %.17g", ca.Fun, cb.Fun)
		}
		if len(ca.X) != len(cb.X) {
			add(ca.Name, "len(x) %d, other %d", len(ca.X), len(cb.X))
		} else {
			for i := range ca.X {
				if !near(ca.X[i], cb.X[i], xTol) {
					add(ca.Name, "x[%d] %.17g, other %.17g", i, ca.X[i], cb.X[i])
					break
				}
			}
		}
		if len(ca.Multipliers) != len(cb.Multipliers) {
			add(ca.Name, "%d multipliers, other %d", len(ca.Multipliers), len(cb.Multipliers))
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
