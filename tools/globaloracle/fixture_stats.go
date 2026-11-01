package main

import "fmt"

// FuncPoint is the value of a benchmark function at one point.
type FuncPoint struct {
	X []Hex `json:"x"`
	F Hex   `json:"f"`
}

// FuncCase describes one function of SciPy's go_benchmark_functions (METHOD
// "funcs"): its class and dimension, bounds, fglob and three evaluations.
type FuncCase struct {
	Name   string      `json:"name"`
	Cls    string      `json:"cls"`
	N      int         `json:"n"`
	Lower  []Hex       `json:"lower"`
	Upper  []Hex       `json:"upper"`
	FGlob  Hex         `json:"fglob"`
	Points []FuncPoint `json:"points"`
}

// StatRun is one seeded run of a global method on a benchmark function.
type StatRun struct {
	Seed    uint64 `json:"seed"`
	Success bool   `json:"success"`
	Nfev    int    `json:"nfev"`
	Nit     int    `json:"nit"`
	Fun     Hex    `json:"fun"`
	X       []Hex  `json:"x"`
	X0      []Hex  `json:"x0,omitempty"`
	Status  int    `json:"status,omitempty"`
}

// StatCase is the runs of one method on one function (METHOD "stats").
type StatCase struct {
	Name   string    `json:"name"`
	Method string    `json:"method"`
	Runs   []StatRun `json:"runs"`
}

func checkFuncs(cases []FuncCase) error {
	seen := map[string]bool{}
	for _, c := range cases {
		if c.Name == "" || seen[c.Name] {
			return fmt.Errorf("fixture: missing or duplicate func name %q", c.Name)
		}
		seen[c.Name] = true
		if c.N < 1 || len(c.Lower) != c.N || len(c.Upper) != c.N || len(c.Points) == 0 {
			return fmt.Errorf("fixture: func %q: bounds or points do not match dimension %d", c.Name, c.N)
		}
		for _, p := range c.Points {
			if len(p.X) != c.N {
				return fmt.Errorf("fixture: func %q: point of length %d", c.Name, len(p.X))
			}
		}
	}
	return nil
}

func checkStats(cases []StatCase) error {
	seen := map[string]bool{}
	for _, c := range cases {
		key := c.Name + "|" + c.Method
		if c.Name == "" || (c.Method != "de" && c.Method != "bh" && c.Method != "direct") || seen[key] {
			return fmt.Errorf("fixture: missing, duplicate or unknown stats case %q (%q)", c.Name, c.Method)
		}
		seen[key] = true
		if len(c.Runs) == 0 {
			return fmt.Errorf("fixture: stats case %q has no runs", key)
		}
		for _, r := range c.Runs {
			if len(r.X) == 0 || (c.Method == "bh" && len(r.X0) != len(r.X)) {
				return fmt.Errorf("fixture: stats case %q seed %d: x or x0 missing", key, r.Seed)
			}
		}
	}
	return nil
}

// diffFuncs compares the function cases of two fixtures exactly.
func diffFuncs(a, b *Fixture, add func(name, format string, args ...any)) {
	byName := map[string]FuncCase{}
	for _, c := range b.Funcs {
		byName[c.Name] = c
	}
	inA := map[string]bool{}
	for _, ca := range a.Funcs {
		inA[ca.Name] = true
		cb, ok := byName[ca.Name]
		if !ok {
			add(ca.Name, "only in the first fixture")
			continue
		}
		if ca.N != cb.N || !sameHexes(ca.Lower, cb.Lower) || !sameHexes(ca.Upper, cb.Upper) ||
			!sameBits(float64(ca.FGlob), float64(cb.FGlob)) || len(ca.Points) != len(cb.Points) {
			add(ca.Name, "dimension, bounds or fglob differ")
			continue
		}
		for i := range ca.Points {
			if !sameHexes(ca.Points[i].X, cb.Points[i].X) || !sameBits(float64(ca.Points[i].F), float64(cb.Points[i].F)) {
				add(ca.Name, "point %d differs", i)
			}
		}
	}
	for _, cb := range b.Funcs {
		if !inA[cb.Name] {
			add(cb.Name, "only in the second fixture")
		}
	}
}

// diffStats compares the stats cases of two fixtures run by run: success,
// nfev, nit and the bit patterns of fun and x.
func diffStats(a, b *Fixture, add func(name, format string, args ...any)) {
	byKey := map[string]StatCase{}
	for _, c := range b.Stats {
		byKey[c.Name+"|"+c.Method] = c
	}
	inA := map[string]bool{}
	for _, ca := range a.Stats {
		key := ca.Name + "|" + ca.Method
		inA[key] = true
		cb, ok := byKey[key]
		if !ok {
			add(key, "only in the first fixture")
			continue
		}
		if len(ca.Runs) != len(cb.Runs) {
			add(key, "%d runs, other %d", len(ca.Runs), len(cb.Runs))
			continue
		}
		for i, ra := range ca.Runs {
			rb := cb.Runs[i]
			if ra.Seed != rb.Seed || ra.Success != rb.Success || ra.Nfev != rb.Nfev || ra.Nit != rb.Nit ||
				ra.Status != rb.Status || !sameBits(float64(ra.Fun), float64(rb.Fun)) || !sameHexes(ra.X, rb.X) {
				add(key, "run %d (seed %d) differs", i, ra.Seed)
			}
		}
	}
	for _, cb := range b.Stats {
		key := cb.Name + "|" + cb.Method
		if !inA[key] {
			add(key, "only in the second fixture")
		}
	}
}
