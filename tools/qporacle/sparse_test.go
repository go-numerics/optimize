package main

import (
	"bytes"
	"strings"
	"testing"
)

// A two-variable, one-row problem in the sparse form: P = I, A = [1 1].
const sparseCaseJSON = `{"cases":[{"name":"s","n":2,"m":1,"p":null,"a":null,` +
	`"p_csc":{"indptr":[0,1,2],"indices":[0,1],"data":[1,1]},` +
	`"a_csc":{"indptr":[0,1,2],"indices":[0,0],"data":[1,1]},` +
	`"q":[0,0],"l":[1],"u":[1],"results":{"s":{"x":[0.5,0.5],"status":"ok"}}}]}`

func TestSparseFixtureDecodes(t *testing.T) {
	f, err := Decode(strings.NewReader(sparseCaseJSON))
	if err != nil {
		t.Fatal(err)
	}
	c := &f.Cases[0]
	if c.Rows() != 1 || c.PCSC == nil || c.ACSC == nil || c.P != nil || c.A != nil {
		t.Errorf("decoded %+v", c)
	}
	var buf bytes.Buffer
	if err := Encode(&buf, f); err != nil {
		t.Fatal(err)
	}
	g, err := Decode(&buf)
	if err != nil {
		t.Fatal(err)
	}
	if d := Diff(f, g, 0); len(d) != 0 {
		t.Errorf("round trip: %v", d)
	}
}

// Every way a sparse case can be wrong is refused; each row changes one thing of
// the valid case above.
func TestSparseFixtureRejects(t *testing.T) {
	for name, tc := range map[string][2]string{
		"indptr too short":     {`"indptr":[0,1,2],"indices":[0,1]`, `"indptr":[0,2],"indices":[0,1]`},
		"indptr not from zero": {`"indptr":[0,1,2],"indices":[0,1]`, `"indptr":[1,1,2],"indices":[0,1]`},
		"indptr overruns":      {`"indptr":[0,1,2],"indices":[0,1]`, `"indptr":[0,9,2],"indices":[0,1]`},
		"indptr decreases":     {`"indptr":[0,1,2],"indices":[0,1]`, `"indptr":[0,2,1],"indices":[0,1]`},
		"indptr negative":      {`"indptr":[0,1,2],"indices":[0,1]`, `"indptr":[0,-1,2],"indices":[0,1]`},
		"indptr huge":          {`"indptr":[0,1,2],"indices":[0,1]`, `"indptr":[0,1,99999999999],"indices":[0,1]`},
		"row out of range":     {`"indptr":[0,1,2],"indices":[0,1]`, `"indptr":[0,1,2],"indices":[0,2]`},
		"row negative":         {`"indptr":[0,1,2],"indices":[0,1]`, `"indptr":[0,1,2],"indices":[0,-1]`},
		"duplicate rows":       {`"indptr":[0,2,2],"indices":[0,1]`, `"indptr":[0,2,2],"indices":[1,1]`},
		"unsorted rows":        {`"indptr":[0,2,2],"indices":[0,1]`, `"indptr":[0,2,2],"indices":[1,0]`},
		"data short":           {`"data":[1,1]},"a_csc"`, `"data":[1]},"a_csc"`},
		"m without sparse a":   {`"m":1,"p":null,"a":null`, `"m":1,"p":null,"a":[[1,1]]`},
		"wrong m":              {`"m":1`, `"m":2`},
		"both forms":           {`"p":null`, `"p":[[1,0],[0,1]]`},
	} {
		in := sparseCaseJSON
		if name == "indptr decreases" || name == "duplicate rows" || name == "unsorted rows" {
			// these need a column with two entries: put both entries of P in column 0
			in = strings.Replace(in, `"p_csc":{"indptr":[0,1,2],"indices":[0,1],"data":[1,1]}`,
				`"p_csc":{"indptr":[0,2,2],"indices":[0,1],"data":[1,1]}`, 1)
			if name == "indptr decreases" {
				in = sparseCaseJSON
			}
		}
		bad := strings.Replace(in, tc[0], tc[1], 1)
		if name == "data short" {
			bad = strings.Replace(in, `"data":[1,1]},"a_csc"`, `"data":[1]},"a_csc"`, 1)
		}
		if bad == in {
			t.Errorf("%s: the mutation did not apply", name)
			continue
		}
		if _, err := Decode(strings.NewReader(bad)); err == nil {
			t.Errorf("%s: accepted", name)
		}
	}
	if _, err := Decode(strings.NewReader(sparseCaseJSON)); err != nil {
		t.Errorf("the control is rejected: %v", err)
	}
}

func TestSparseFixtureDiff(t *testing.T) {
	a, _ := Decode(strings.NewReader(sparseCaseJSON))
	for name, mut := range map[string]func(*Case){
		"data":    func(c *Case) { c.ACSC.Data[0] = 2 },
		"indices": func(c *Case) { c.ACSC.Indices = []int{0, 0}; c.ACSC.Indptr = []int{0, 2, 2} },
		"m":       func(c *Case) { c.M = 2 },
		"p":       func(c *Case) { c.PCSC = nil; c.P = [][]float64{{1, 0}, {0, 1}} },
	} {
		b, _ := Decode(strings.NewReader(sparseCaseJSON))
		mut(&b.Cases[0])
		d := Diff(a, b, 0)
		if len(d) != 1 || d[0].What != "the problem data differ" {
			t.Errorf("%s: %v", name, d)
		}
	}
	if d := Diff(a, a, 0); len(d) != 0 {
		t.Errorf("identical fixtures differ: %v", d)
	}
}
