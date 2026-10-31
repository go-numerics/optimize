package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"sort"

	"github.com/spf13/cobra"
)

// Event is one evaluation ("f", "g", "c0", "j0", ...) or iteration ("it") of a
// traced run. In JSON it is the pair [tag, x].
type Event struct {
	Tag string
	X   []float64
}

// MarshalJSON writes the [tag, x] pair.
func (e Event) MarshalJSON() ([]byte, error) { return json.Marshal([]any{e.Tag, e.X}) }

// UnmarshalJSON reads the [tag, x] pair.
func (e *Event) UnmarshalJSON(b []byte) error {
	var raw []json.RawMessage
	if err := json.Unmarshal(b, &raw); err != nil {
		return err
	}
	if len(raw) != 2 {
		return fmt.Errorf("event: want [tag, x], got %d elements", len(raw))
	}
	if err := json.Unmarshal(raw[0], &e.Tag); err != nil {
		return err
	}
	return json.Unmarshal(raw[1], &e.X)
}

// Trace is the content of a trace file: the events of one run.
type Trace struct {
	Case   string  `json:"case"`
	Events []Event `json:"events"`
	// The rest describes the SciPy side and is ignored by tracediff.
	Method  string          `json:"method,omitempty"`
	Scipy   string          `json:"scipy,omitempty"`
	Numpy   string          `json:"numpy,omitempty"`
	Machine string          `json:"machine,omitempty"`
	Result  json.RawMessage `json:"result,omitempty"`
}

// DecodeTrace reads a trace and rejects an empty one.
func DecodeTrace(r io.Reader) (*Trace, error) {
	var t Trace
	if err := json.NewDecoder(r).Decode(&t); err != nil {
		return nil, fmt.Errorf("trace: %w", err)
	}
	if len(t.Events) == 0 {
		return nil, fmt.Errorf("trace: no events")
	}
	return &t, nil
}

// Divergence is where the stream of one tag first parts between two traces.
type Divergence struct {
	Tag string
	// Index is the position in the stream of that tag (0 = its first event).
	Index int
	// Iter is the number of "it" events before it in the first trace.
	Iter int
	// MaxRel is the largest relative difference of x (+Inf when one stream is
	// shorter than the other).
	MaxRel float64
}

func relDiff(a, b float64) float64 {
	if a == b || (math.IsNaN(a) && math.IsNaN(b)) {
		return 0
	}
	return math.Abs(a-b) / math.Max(math.Abs(a), math.Abs(b))
}

func xDiff(a, b []float64) float64 {
	if len(a) != len(b) {
		return math.Inf(1)
	}
	m := 0.0
	for i := range a {
		m = math.Max(m, relDiff(a[i], b[i]))
	}
	return m
}

// Compare compares the traces tag by tag: SciPy evaluates the gradient and
// the constraint Jacobians in a different order from ours within an
// iteration, so only the stream of each tag is compared. It returns, for each
// tag whose stream differs bitwise, the first event that does, and for each
// tag with a difference above tol (relative, or a different length), the first
// event that does; both sorted by iteration.
func Compare(a, b []Event, tol float64) (bitwise, beyond []Divergence) {
	type stream struct {
		x    [][]float64
		iter []int
	}
	split := func(ev []Event) map[string]*stream {
		out := map[string]*stream{}
		iter := 0
		for _, e := range ev {
			s := out[e.Tag]
			if s == nil {
				s = &stream{}
				out[e.Tag] = s
			}
			s.x, s.iter = append(s.x, e.X), append(s.iter, iter)
			if e.Tag == "it" {
				iter++
			}
		}
		return out
	}
	sa, sb := split(a), split(b)
	tags := map[string]bool{}
	for t := range sa {
		tags[t] = true
	}
	for t := range sb {
		tags[t] = true
	}
	empty := &stream{}
	for tag := range tags {
		x, y := sa[tag], sb[tag]
		if x == nil {
			x = empty
		}
		if y == nil {
			y = empty
		}
		n := min(len(x.x), len(y.x))
		var first, firstTol *Divergence
		for i := 0; i < n && firstTol == nil; i++ {
			d := xDiff(x.x[i], y.x[i])
			if d != 0 && first == nil {
				first = &Divergence{tag, i, x.iter[i], d}
			}
			if d > tol {
				firstTol = &Divergence{tag, i, x.iter[i], d}
			}
		}
		if len(x.x) != len(y.x) && firstTol == nil {
			it := 0
			if n < len(x.iter) {
				it = x.iter[n]
			} else if n > 0 {
				it = x.iter[n-1]
			}
			firstTol = &Divergence{tag, n, it, math.Inf(1)}
			if first == nil {
				first = firstTol
			}
		}
		if first != nil {
			bitwise = append(bitwise, *first)
		}
		if firstTol != nil {
			beyond = append(beyond, *firstTol)
		}
	}
	order := func(d []Divergence) {
		sort.Slice(d, func(i, j int) bool {
			if d[i].Iter != d[j].Iter {
				return d[i].Iter < d[j].Iter
			}
			return d[i].Tag < d[j].Tag
		})
	}
	order(bitwise)
	order(beyond)
	return bitwise, beyond
}

func tracediffCmd() *cobra.Command {
	var tol float64
	var show []string
	var showIndex int
	var quiet bool
	cmd := &cobra.Command{
		Use:   "tracediff first.json second.json",
		Short: "Find the first evaluation where two traces part (exit 1 if they do)",
		Long: "tracediff compares the traces written by `slsqporacle trace` (SciPy) and by\n" +
			"`go test -run TestTraceDump` (Go), one stream per tag (f, g, c0, j0, it, ...),\n" +
			"and prints, per tag, the first event that differs bitwise and the first that\n" +
			"differs by more than --tol (relative), earliest iteration first.",
		SilenceUsage: true,
		Args:         cobra.ExactArgs(2),
		RunE: func(cmd *cobra.Command, args []string) error {
			var tr [2]*Trace
			for i, p := range args {
				b, err := os.ReadFile(p)
				if err != nil {
					return err
				}
				if tr[i], err = DecodeTrace(bytes.NewReader(b)); err != nil {
					return fmt.Errorf("%s: %w", p, err)
				}
			}
			w := cmd.OutOrStdout()
			bit, beyond := Compare(tr[0].Events, tr[1].Events, tol)
			fmt.Fprintf(w, "%s: %d events; %s: %d events\n", args[0], len(tr[0].Events), args[1], len(tr[1].Events))
			if len(bit) == 0 {
				fmt.Fprintln(w, "identical")
				return nil
			}
			for _, d := range bit {
				if quiet {
					break
				}
				fmt.Fprintf(w, "bitwise: tag %q event %d (after %d iterations), max relative difference %.3g\n", d.Tag, d.Index, d.Iter, d.MaxRel)
			}
			for _, tag := range show {
				for i, t := range tr {
					k := 0
					for _, e := range t.Events {
						if e.Tag == tag {
							if k == showIndex {
								fmt.Fprintf(w, "%s[%d] in %s: %v\n", tag, k, args[i], e.X)
							}
							k++
						}
					}
				}
			}
			for _, d := range beyond {
				if quiet {
					break
				}
				fmt.Fprintf(w, "above %.3g: tag %q event %d (after %d iterations), max relative difference %.3g\n", tol, d.Tag, d.Index, d.Iter, d.MaxRel)
			}
			if len(beyond) == 0 {
				fmt.Fprintf(w, "no difference above %.3g\n", tol)
			}
			return ErrDisagree
		},
	}
	cmd.Flags().Float64Var(&tol, "tol", 1e-12, "relative tolerance of the second report")
	cmd.Flags().StringSliceVar(&show, "show", nil, "tags whose event number --index is printed from both traces")
	cmd.Flags().IntVar(&showIndex, "index", 0, "event number of --show")
	cmd.Flags().BoolVar(&quiet, "quiet", false, "print only the --show events and the exit status")
	return cmd
}

func traceCmd() *cobra.Command {
	var out, platform, method string
	var blasVerbose bool
	cmd := &cobra.Command{
		Use:   "trace case",
		Short: "Record every evaluation and iteration of one SciPy run (e.g. HS21/jac)",
		Long: "trace runs one case of hs.py (name as in the fixtures: HS21/jac, HS48/fd)\n" +
			"with SciPy and writes its objective, gradient and constraint evaluations and\n" +
			"its iterations, in order, to the --output file.",
		SilenceUsage: true,
		Args:         cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			if method != "SLSQP" && method != "trust-constr" {
				return fmt.Errorf("--method must be SLSQP or trust-constr, not %q", method)
			}
			pre := fmt.Sprintf("METHOD = %q\nTRACE = %q\n", method, args[0])
			if blasVerbose {
				// OpenBLAS prints the core (kernel set) it selected at load time.
				pre = "import os\nos.environ['OPENBLAS_VERBOSE'] = '2'\n" + pre
			}
			src := append([]byte(pre), script...)
			var buf bytes.Buffer
			if err := runner(command(platform), bytes.NewReader(src), &buf, cmd.ErrOrStderr()); err != nil {
				return fmt.Errorf("oracle: %w", err)
			}
			t, err := DecodeTrace(&buf)
			if err != nil {
				return fmt.Errorf("oracle output: %w", err)
			}
			b, _ := json.Marshal(t) // cannot fail: t was just decoded from JSON
			if err := os.WriteFile(out, b, 0o644); err != nil {
				return err
			}
			fmt.Fprintf(cmd.OutOrStdout(), "wrote %d events of %s (SciPy %s, %s) to %s\n", len(t.Events), t.Case, t.Scipy, t.Machine, out)
			return nil
		},
	}
	cmd.Flags().BoolVar(&blasVerbose, "blas-verbose", false, "set OPENBLAS_VERBOSE=2 so that stderr names the OpenBLAS core in use")
	cmd.Flags().StringVarP(&out, "output", "o", "trace.json", "trace file to write")
	cmd.Flags().StringVar(&method, "method", "trust-constr", "minimize method: SLSQP or trust-constr")
	cmd.Flags().StringVar(&platform, "platform", "",
		"run SciPy in a container of this platform (e.g. linux/amd64) instead of on the host")
	return cmd
}
