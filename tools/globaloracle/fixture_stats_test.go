package main

import (
	"bytes"
	"strings"
	"testing"
)

func statFixture() *Fixture {
	return &Fixture{
		Method: "stats", Scipy: "1.18.1", Numpy: "2.5.4", Python: "3.12", Machine: "x86_64", System: "Linux",
		Sources: map[string]string{"go_benchmark.py": "00"},
		Funcs: []FuncCase{{Name: "Sphere/2", Cls: "Sphere", N: 2, Lower: []Hex{-5, -5}, Upper: []Hex{5, 5}, FGlob: 0,
			Points: []FuncPoint{{X: []Hex{1, 2}, F: 5}}}},
		Stats: []StatCase{{Name: "Sphere/2", Method: "de", Runs: []StatRun{
			{Seed: 0, Success: true, Nfev: 10, Nit: 2, Fun: 0.5, X: []Hex{0.1, 0.2}},
			{Seed: 1, Success: false, Nfev: 12, Nit: 3, Fun: 1.5, X: []Hex{0.3, 0.2}}}},
			{Name: "Sphere/2", Method: "bh", Runs: []StatRun{{Seed: 0, X: []Hex{1, 2}, X0: []Hex{3, 4}}}}},
	}
}

func roundTrip(t *testing.T, f *Fixture) (*Fixture, error) {
	t.Helper()
	var buf bytes.Buffer
	if err := Encode(&buf, f); err != nil {
		t.Fatal(err)
	}
	return Decode(&buf)
}

func TestFuncsAndStatsRoundTripAndDiff(t *testing.T) {
	a, err := roundTrip(t, statFixture())
	if err != nil {
		t.Fatal(err)
	}
	b, _ := roundTrip(t, statFixture())
	if d := Diff(a, b); len(d) != 0 {
		t.Fatalf("identical fixtures disagree: %v", d)
	}
	// Each planted difference is reported.
	for name, mutate := range map[string]func(*Fixture){
		"fglob":   func(f *Fixture) { f.Funcs[0].FGlob = 1 },
		"point":   func(f *Fixture) { f.Funcs[0].Points[0].F = 6 },
		"bounds":  func(f *Fixture) { f.Funcs[0].Upper[0] = 6 },
		"success": func(f *Fixture) { f.Stats[0].Runs[0].Success = false },
		"nfev":    func(f *Fixture) { f.Stats[0].Runs[1].Nfev++ },
		"fun":     func(f *Fixture) { f.Stats[0].Runs[1].Fun = 2 },
		"x":       func(f *Fixture) { f.Stats[1].Runs[0].X[0] = 9 },
		"runs":    func(f *Fixture) { f.Stats[0].Runs = f.Stats[0].Runs[:1] },
		"missing": func(f *Fixture) { f.Stats = f.Stats[:1] },
		"nofunc":  func(f *Fixture) { f.Funcs = nil },
	} {
		c, _ := roundTrip(t, statFixture())
		mutate(c)
		if len(Diff(a, c)) == 0 || len(Diff(c, a)) == 0 {
			t.Errorf("%s: difference not reported", name)
		}
	}
	extra, _ := roundTrip(t, statFixture())
	extra.Funcs = append(extra.Funcs, FuncCase{Name: "Other/2", N: 1, Lower: []Hex{0}, Upper: []Hex{1}, Points: []FuncPoint{{X: []Hex{0}}}})
	if len(Diff(a, extra)) == 0 || len(Diff(extra, a)) == 0 {
		t.Error("an extra function is not reported")
	}
}

func TestFuncsAndStatsValidation(t *testing.T) {
	for name, mutate := range map[string]func(*Fixture){
		"duplicate func": func(f *Fixture) { f.Funcs = append(f.Funcs, f.Funcs[0]) },
		"nameless func":  func(f *Fixture) { f.Funcs[0].Name = "" },
		"bounds length":  func(f *Fixture) { f.Funcs[0].Lower = f.Funcs[0].Lower[:1] },
		"no points":      func(f *Fixture) { f.Funcs[0].Points = nil },
		"point length":   func(f *Fixture) { f.Funcs[0].Points[0].X = []Hex{1} },
		"duplicate stat": func(f *Fixture) { f.Stats = append(f.Stats, f.Stats[0]) },
		"unknown method": func(f *Fixture) { f.Stats[0].Method = "sa" },
		"nameless stat":  func(f *Fixture) { f.Stats[0].Name = "" },
		"no runs":        func(f *Fixture) { f.Stats[0].Runs = nil },
		"no x":           func(f *Fixture) { f.Stats[0].Runs[0].X = nil },
		"bh without x0":  func(f *Fixture) { f.Stats[1].Runs[0].X0 = nil },
	} {
		f := statFixture()
		mutate(f)
		var buf bytes.Buffer
		if err := Encode(&buf, f); err != nil {
			t.Fatal(err)
		}
		if _, err := Decode(&buf); err == nil || !strings.Contains(err.Error(), "fixture") {
			t.Errorf("%s: accepted (%v)", name, err)
		}
	}
}

func TestScriptFor(t *testing.T) {
	if string(scriptFor("funcs")) != string(gobScript) || string(scriptFor("stats")) != string(gobScript) {
		t.Error("funcs and stats must use the benchmark-function oracle")
	}
	for _, m := range []string{"direct", "de", "bh"} {
		if string(scriptFor(m)) != string(script) {
			t.Errorf("%s must use global.py", m)
		}
	}
	for _, m := range []string{"funcs", "stats"} {
		if !strings.Contains(string(gobScript), `METHOD == "`+m+`"`) {
			t.Errorf("gob.py has no branch for %s", m)
		}
	}
}
