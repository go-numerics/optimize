package main

import (
	"bytes"
	"strings"
	"testing"
)

func deCase(name string) DECase {
	h := Hex(1.5)
	return DECase{
		Name: name, Problem: "booth", Dim: 2, Lower: []float64{0, 0}, Upper: []float64{1, 1},
		Kind: "rng", Seed: 1, X: []Hex{1, 2}, Fun: &h, Nfev: 10, Nit: 2, Success: true, Message: "ok",
		Population: [][]Hex{{1, 2}, {3, 4}}, Energies: []Hex{5, 6},
	}
}

func TestDEFixtureRoundTrip(t *testing.T) {
	f := &Fixture{Method: "de", DE: []DECase{deCase("a"), deCase("b")},
		Reductions: []Reduction{{Name: "r", Values: []Hex{1, 2}, Mean: 1.5, Std: 0.5}}}
	var buf bytes.Buffer
	if err := Encode(&buf, f); err != nil {
		t.Fatal(err)
	}
	g, err := Decode(&buf)
	if err != nil || len(g.DE) != 2 || len(g.Reductions) != 1 {
		t.Fatalf("%v %+v", err, g)
	}
	if d := Diff(f, g); len(d) != 0 {
		t.Errorf("round trip differs: %v", d)
	}
}

func TestCheckDE(t *testing.T) {
	bad := func(mut func(c *DECase)) error {
		c := deCase("a")
		mut(&c)
		return checkDE([]DECase{c})
	}
	for name, mut := range map[string]func(c *DECase){
		"name":       func(c *DECase) { c.Name = "" },
		"dim":        func(c *DECase) { c.Dim = 3 },
		"kind":       func(c *DECase) { c.Kind = "x" },
		"no x":       func(c *DECase) { c.X = nil },
		"energies":   func(c *DECase) { c.Energies = c.Energies[:1] },
		"row length": func(c *DECase) { c.Population[0] = []Hex{1} },
	} {
		if bad(mut) == nil {
			t.Errorf("%s: accepted", name)
		}
	}
	e := "boom"
	if err := bad(func(c *DECase) { c.Error = &e; c.X = nil; c.Population = nil }); err != nil {
		t.Error(err)
	}
	if err := checkDE([]DECase{deCase("a"), deCase("a")}); err == nil || !strings.Contains(err.Error(), "duplicate") {
		t.Error(err)
	}
}

func TestDiffDE(t *testing.T) {
	a := &Fixture{DE: []DECase{deCase("same"), deCase("msg"), deCase("nfev"), deCase("x"), deCase("pop"), deCase("err"), deCase("only-a")},
		Reductions: []Reduction{{Name: "r", Values: []Hex{1}, Mean: 1, Std: 0}}}
	b := &Fixture{DE: []DECase{deCase("same"), deCase("msg"), deCase("nfev"), deCase("x"), deCase("pop"), deCase("err"), deCase("only-b")},
		Reductions: []Reduction{{Name: "r", Values: []Hex{1}, Mean: 2, Std: 0}, {Name: "s"}}}
	b.DE[1].Message = "other"
	b.DE[2].Nfev = 11
	b.DE[3].X = []Hex{1, 3}
	b.DE[4].Population[1] = []Hex{3, 5}
	e := "boom"
	b.DE[5].Error = &e
	d := Diff(a, b)
	got := map[string]bool{}
	for _, x := range d {
		got[x.Case] = true
	}
	for _, want := range []string{"msg", "nfev", "x", "pop", "err", "only-a", "only-b", "reduction r", "reductions"} {
		if !got[want] {
			t.Errorf("missing disagreement %q in %v", want, d)
		}
	}
	if got["same"] {
		t.Error("identical case reported")
	}
	// both sides in error: equal messages agree, then skipped
	c := deCase("e")
	c.Error = &e
	if len(Diff(&Fixture{DE: []DECase{c}}, &Fixture{DE: []DECase{c}})) != 0 {
		t.Error("errors not skipped")
	}
}
