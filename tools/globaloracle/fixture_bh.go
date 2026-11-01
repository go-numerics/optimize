package main

import "fmt"

// checkBH validates the basin-hopping cases of a fixture: unique names,
// known seed kinds, results and traces that match the dimension.
func checkBH(cases []BHCase) error {
	seen := map[string]bool{}
	for _, c := range cases {
		if c.Name == "" || seen[c.Name] {
			return fmt.Errorf("fixture: missing or duplicate bh case name %q", c.Name)
		}
		seen[c.Name] = true
		if c.Kind != "rng" && c.Kind != "seed" {
			return fmt.Errorf("fixture: bh case %q: seed kind %q", c.Name, c.Kind)
		}
		if c.Error != nil {
			continue
		}
		if len(c.X) != len(c.X0) || c.Fun == nil || len(c.Trace) == 0 {
			return fmt.Errorf("fixture: bh case %q: x, fun or trace missing", c.Name)
		}
		for _, s := range c.Trace {
			if len(s.X) != len(c.X0) {
				return fmt.Errorf("fixture: bh case %q: trace point of length %d", c.Name, len(s.X))
			}
		}
	}
	return nil
}

// diffBH compares the basin-hopping cases of two fixtures exactly: the error,
// the counts, the message, x, fun and every step of the trace.
func diffBH(a, b *Fixture, add func(name, format string, args ...any)) {
	byName := map[string]BHCase{}
	for _, c := range b.BH {
		byName[c.Name] = c
	}
	inA := map[string]bool{}
	for _, ca := range a.BH {
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
		if ca.Nit != cb.Nit || ca.Nfev != cb.Nfev || ca.Njev != cb.Njev || ca.Nhev != cb.Nhev ||
			ca.Failures != cb.Failures || ca.Success != cb.Success || ca.Message != cb.Message {
			add(ca.Name, "counts, success or message differ")
		}
		if !sameHexes(ca.X, cb.X) || !sameBits(float64(*ca.Fun), float64(*cb.Fun)) {
			add(ca.Name, "x or fun differ")
		}
		same := len(ca.Trace) == len(cb.Trace)
		for i := 0; same && i < len(ca.Trace); i++ {
			same = ca.Trace[i].Accept == cb.Trace[i].Accept && sameHexes(ca.Trace[i].X, cb.Trace[i].X) &&
				sameBits(float64(ca.Trace[i].F), float64(cb.Trace[i].F))
			if !same {
				add(ca.Name, "trace differs at step %d", i)
			}
		}
		if !same && len(ca.Trace) != len(cb.Trace) {
			add(ca.Name, "trace length %d, other %d", len(ca.Trace), len(cb.Trace))
		}
	}
	for _, cb := range b.BH {
		if !inA[cb.Name] {
			add(cb.Name, "only in the second fixture")
		}
	}
}
