package main

import (
	"bytes"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

func TestParseLoad(t *testing.T) {
	for in, want := range map[string]float64{
		"{ 4.61 5.02 6.10 }\n":           4.61,
		"0.52 0.48 0.40 1/123 4567\n":    0.52,
		"load averages: 12.03 9.79 9.12": 12.03,
	} {
		if got, err := parseLoad(in); err != nil || got != want {
			t.Errorf("parseLoad(%q) = %v, %v", in, got, err)
		}
	}
	if _, err := parseLoad("{ }"); err == nil {
		t.Error("no number accepted")
	}
}

func TestParseGoBench(t *testing.T) {
	in := `goos: darwin
BenchmarkSLSQPHS71   	   42607	      5576 ns/op	    5360 B/op	      38 allocs/op
BenchmarkSLSQPQuad50-8 	      50	   4537957 ns/op
BenchmarkBroken  12  notanumber ns/op
PASS
`
	got := parseGoBench(strings.NewReader(in))
	ns := func(v float64) float64 { return v * 1e-9 }
	if len(got) != 2 || got["SLSQPHS71"] != ns(5576) || got["SLSQPQuad50"] != ns(4537957) {
		t.Errorf("parseGoBench = %v", got)
	}
}

func TestMedianAndHuman(t *testing.T) {
	if median(nil) != 0 || median([]float64{3, 1, 2}) != 2 || median([]float64{4, 1, 3, 2}) != 2.5 {
		t.Error("median")
	}
	if human(5.59e-6) != "5.59 µs" || human(3.35e-3) != "3.35 ms" {
		t.Errorf("human: %q %q", human(5.59e-6), human(3.35e-3))
	}
}

func runCmd(t *testing.T, args ...string) (string, string, error) {
	t.Helper()
	cmd := RootCmd()
	var out, errb bytes.Buffer
	cmd.SetArgs(args)
	cmd.SetOut(&out)
	cmd.SetErr(&errb)
	err := cmd.Execute()
	return out.String(), errb.String(), err
}

// withLoad injects a constant load reading.
func withLoad(t *testing.T, v float64, err error) {
	old := loadAvg
	t.Cleanup(func() { loadAvg = old })
	loadAvg = func() (float64, error) { return v, err }
}

// TestRefusesUnderLoad: above --max-load nothing is built or run.
func TestRefusesUnderLoad(t *testing.T) {
	withLoad(t, 12.3, nil)
	ran := false
	old := execRun
	t.Cleanup(func() { execRun = old })
	execRun = func(*exec.Cmd) error { ran = true; return nil }
	_, _, err := runCmd(t, "--max-load", "8")
	if err == nil || !strings.Contains(err.Error(), "deferred") || ran {
		t.Errorf("err %v, ran %v", err, ran)
	}
	withLoad(t, 0, errors.New("no sysctl"))
	if _, _, err := runCmd(t); err == nil {
		t.Error("load error ignored")
	}
}

// fakeExec simulates go test -c, the test binary and uv. fail names the
// stage that fails; py is SciPy's JSON output.
func fakeExec(t *testing.T, fail, py string) {
	old := execRun
	t.Cleanup(func() { execRun = old })
	execRun = func(c *exec.Cmd) error {
		name := filepath.Base(c.Path)
		switch {
		case name == "go":
			if fail == "build" {
				return errors.New("build failed")
			}
		case strings.HasSuffix(name, ".test"):
			if fail == "go" {
				return errors.New("bench failed")
			}
			for _, n := range benchNames {
				fmt.Fprintf(c.Stdout, "Benchmark%s \t 100\t 1000 ns/op\n", n)
			}
		default: // uv
			if fail == "scipy" {
				return errors.New("uv failed")
			}
			fmt.Fprint(c.Stdout, py)
		}
		return nil
	}
}

func scipyJSON(skip string) string {
	var parts []string
	for _, n := range benchNames {
		if n != skip {
			parts = append(parts, fmt.Sprintf("%q: 2e-5", n))
		}
	}
	return "{" + strings.Join(parts, ", ") + "}"
}

func TestFullRunSimulated(t *testing.T) {
	withLoad(t, 1.5, nil)
	fakeExec(t, "", scipyJSON(""))
	out, log, err := runCmd(t, "--rounds", "3", "--repo", t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out, "Medians of 3 interleaved rounds") || !strings.Contains(out, "| SLSQPHS71 | 1 µs | 20 µs | 20.0× |") {
		t.Errorf("table:\n%s", out)
	}
	if strings.Count(log, "round ") != 3 {
		t.Errorf("log:\n%s", log)
	}
	for stage, py := range map[string]string{"build": scipyJSON(""), "go": scipyJSON(""), "scipy": scipyJSON(""),
		"bad json": "not json", "missing": scipyJSON("SLSQPHS38")} {
		fail := stage
		if stage == "bad json" || stage == "missing" {
			fail = ""
		}
		fakeExec(t, fail, py)
		if _, _, err := runCmd(t, "--rounds", "1", "--repo", t.TempDir()); err == nil {
			t.Errorf("%s: no error", stage)
		}
	}
}

func TestMainExitsOnError(t *testing.T) {
	withLoad(t, 99, nil)
	code := -1
	oldExit, oldArgs := osExit, os.Args
	defer func() { osExit, os.Args = oldExit, oldArgs }()
	osExit = func(c int) { code = c }
	os.Args = []string{"slsqpbench"}
	main()
	if code != 1 {
		t.Errorf("exit code %d", code)
	}
}

func TestLoadAvgReadsTheHost(t *testing.T) {
	if v, err := loadAvg(); err != nil || v < 0 {
		t.Skipf("no load average on this host: %v", err)
	}
}
