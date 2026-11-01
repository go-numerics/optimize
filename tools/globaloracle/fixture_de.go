package main

import "fmt"

// checkDE validates the differential-evolution cases of a fixture: unique
// non-empty names, bounds and results that match the dimension, known seed
// kinds.
func checkDE(cases []DECase) error {
	seen := map[string]bool{}
	for _, c := range cases {
		if c.Name == "" || seen[c.Name] {
			return fmt.Errorf("fixture: missing or duplicate de case name %q", c.Name)
		}
		seen[c.Name] = true
		if c.Dim < 1 || len(c.Lower) != c.Dim || len(c.Upper) != c.Dim {
			return fmt.Errorf("fixture: de case %q: bounds do not match dimension %d", c.Name, c.Dim)
		}
		if c.Kind != "rng" && c.Kind != "seed" {
			return fmt.Errorf("fixture: de case %q: seed kind %q", c.Name, c.Kind)
		}
		if c.Error != nil {
			continue
		}
		if len(c.X) != c.Dim || c.Fun == nil || len(c.Population) == 0 || len(c.Energies) != len(c.Population) {
			return fmt.Errorf("fixture: de case %q: x, fun or population missing", c.Name)
		}
		for _, row := range c.Population {
			if len(row) != c.Dim {
				return fmt.Errorf("fixture: de case %q: population row of length %d, dimension %d", c.Name, len(row), c.Dim)
			}
		}
	}
	return nil
}

func sameHexes(a, b []Hex) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if !sameBits(float64(a[i]), float64(b[i])) {
			return false
		}
	}
	return true
}

// diffDE compares the differential-evolution cases of two fixtures exactly:
// the error, success, message, nfev, nit and the bits of x, fun, the final
// population and its energies. Cases on one side only are reported.
func diffDE(a, b *Fixture, add func(name, format string, args ...any)) {
	byName := map[string]DECase{}
	for _, c := range b.DE {
		byName[c.Name] = c
	}
	inA := map[string]bool{}
	for _, ca := range a.DE {
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
		if ca.Success != cb.Success || ca.Message != cb.Message {
			add(ca.Name, "success %v %q, other %v %q", ca.Success, ca.Message, cb.Success, cb.Message)
		}
		if ca.Nit != cb.Nit || ca.Nfev != cb.Nfev {
			add(ca.Name, "nit/nfev %d/%d, other %d/%d", ca.Nit, ca.Nfev, cb.Nit, cb.Nfev)
		}
		if !sameBits(float64(*ca.Fun), float64(*cb.Fun)) || !sameHexes(ca.X, cb.X) {
			add(ca.Name, "x or fun differ")
		}
		same := len(ca.Population) == len(cb.Population) && sameHexes(ca.Energies, cb.Energies)
		for i := 0; same && i < len(ca.Population); i++ {
			same = sameHexes(ca.Population[i], cb.Population[i])
		}
		if !same {
			add(ca.Name, "population or energies differ")
		}
	}
	for _, cb := range b.DE {
		if !inA[cb.Name] {
			add(cb.Name, "only in the second fixture")
		}
	}
	rb := map[string]Reduction{}
	for _, r := range b.Reductions {
		rb[r.Name] = r
	}
	for _, ra := range a.Reductions {
		r, ok := rb[ra.Name]
		if !ok || !sameHexes(ra.Values, r.Values) || !sameBits(float64(ra.Mean), float64(r.Mean)) ||
			!sameBits(float64(ra.Std), float64(r.Std)) {
			add("reduction "+ra.Name, "differs or missing in the second fixture")
		}
	}
	if len(a.Reductions) != len(b.Reductions) {
		add("reductions", "%d against %d", len(a.Reductions), len(b.Reductions))
	}
}
