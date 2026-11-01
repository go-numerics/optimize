package main

import (
	"bytes"
	"errors"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func fakeGet(t *testing.T, body string) func(string) (io.ReadCloser, error) {
	return func(url string) (io.ReadCloser, error) {
		if !strings.HasPrefix(url, mmURL) || !strings.HasSuffix(url, ".mat") {
			t.Errorf("unexpected URL %s", url)
		}
		return io.NopCloser(strings.NewReader(body)), nil
	}
}

func TestMMTiersAndNames(t *testing.T) {
	for tier, want := range map[string]int{"tiny": 16, "small": 17, "all": 33} {
		names, err := mmNames(tier)
		if err != nil || len(names) != want {
			t.Errorf("%s: %d names, %v", tier, len(names), err)
		}
		for _, n := range names {
			if _, ok := mmSums[n]; !ok || !mmSafeName(n) {
				t.Errorf("%s: no pinned sum or unsafe name", n)
			}
		}
	}
	if _, err := mmNames("huge"); err == nil {
		t.Error("unknown tier accepted")
	}
	for _, bad := range []string{"", "../x", "a b", "a;b", strings.Repeat("A", 33), "a/b"} {
		if mmSafeName(bad) {
			t.Errorf("%q accepted", bad)
		}
	}
}

func TestMMFetch(t *testing.T) {
	old := httpGet
	defer func() { httpGet = old }()
	dir := t.TempDir()
	// A body that does not match the pinned sum is refused and not written.
	httpGet = fakeGet(t, "not a mat file")
	if _, err := fetchMM([]string{"HS21"}, dir, 1<<20, false); err == nil || !strings.Contains(err.Error(), "sha256") {
		t.Errorf("sum mismatch: %v", err)
	}
	if _, err := os.Stat(filepath.Join(dir, "HS21.mat")); err == nil {
		t.Error("an unverified file was written")
	}
	// Record mode returns the sums and writes the files.
	seen, err := fetchMM([]string{"HS21"}, dir, 1<<20, true)
	if err != nil || len(seen["HS21"]) != 64 {
		t.Fatalf("record: %v %v", err, seen)
	}
	if _, err := fetchMM([]string{"HS21"}, dir, 4, true); err == nil || !strings.Contains(err.Error(), "max-bytes") {
		t.Errorf("oversize: %v", err)
	}
	if _, err := fetchMM([]string{"../evil"}, dir, 1<<20, true); err == nil {
		t.Error("unsafe name accepted")
	}
	httpGet = func(string) (io.ReadCloser, error) { return nil, errors.New("offline") }
	if _, err := fetchMM([]string{"HS21"}, dir, 1<<20, true); err == nil {
		t.Error("network error swallowed")
	}
	if _, err := fetchMM([]string{"HS21"}, filepath.Join(dir, "HS21.mat", "sub"), 1<<20, true); err == nil {
		t.Error("unwritable directory accepted")
	}
	// The command: record prints lines, a verified fetch with a wrong body fails.
	httpGet = fakeGet(t, "x")
	out, err := run(t, "mm", "fetch", "--tier", "tiny", "--dir", dir, "--record")
	if err != nil || strings.Count(out, "\n") != 16 {
		t.Errorf("record command: %v %q", err, out)
	}
	if _, err := run(t, "mm", "fetch", "--tier", "tiny", "--dir", dir); err == nil {
		t.Error("verified fetch of a wrong body succeeded")
	}
	if _, err := run(t, "mm", "fetch", "--tier", "nope"); err == nil {
		t.Error("unknown tier accepted")
	}
}

func TestMMConvert(t *testing.T) {
	raw, err := os.ReadFile("../../testdata/qp/mm/tiny.json")
	if err != nil {
		t.Fatal(err)
	}
	old := runner
	defer func() { runner = old }()
	var head string
	runner = func(argv []string, stdin io.Reader, stdout, _ io.Writer) error {
		in, _ := io.ReadAll(stdin)
		head, _, _ = strings.Cut(string(in), "\n")
		_, err := stdout.Write(raw)
		return err
	}
	dir := t.TempDir()
	dst := filepath.Join(dir, "o.json")
	if _, err := run(t, "mm", "convert", "--tier", "tiny", "--dir", dir, "-o", dst); err == nil {
		t.Error("missing MAT files accepted")
	}
	names, _ := mmNames("tiny")
	for _, n := range names {
		if err := os.WriteFile(filepath.Join(dir, n+".mat"), nil, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	out, err := run(t, "mm", "convert", "--tier", "tiny", "--dir", dir, "-o", dst)
	if err != nil || !strings.Contains(out, "wrote 16 cases") || head != `QP_SET = "mm"` {
		t.Fatalf("%v %q %q", err, out, head)
	}
	runner = func([]string, io.Reader, io.Writer, io.Writer) error { return errors.New("boom") }
	if _, err := run(t, "mm", "convert", "--tier", "tiny", "--dir", dir, "-o", dst); err == nil {
		t.Error("runner failure swallowed")
	}
	runner = func(_ []string, _ io.Reader, w io.Writer, _ io.Writer) error {
		_, err := w.Write([]byte("garbage"))
		return err
	}
	if _, err := run(t, "mm", "convert", "--tier", "tiny", "--dir", dir, "-o", dst); err == nil {
		t.Error("garbage output accepted")
	}
	fx, err := Decode(bytes.NewReader(raw))
	if err != nil {
		t.Fatal(err)
	}
	fx.OsqpCore = "0.6.3"
	var badBuf bytes.Buffer
	if err := Encode(&badBuf, fx); err != nil {
		t.Fatal(err)
	}
	bad := badBuf.Bytes()
	runner = func(_ []string, _ io.Reader, w io.Writer, _ io.Writer) error { _, err := w.Write(bad); return err }
	if _, err := run(t, "mm", "convert", "--tier", "tiny", "--dir", dir, "-o", dst); err == nil {
		t.Error("wrong core accepted")
	}
	runner = func(_ []string, _ io.Reader, w io.Writer, _ io.Writer) error { _, err := w.Write(raw); return err }
	if _, err := run(t, "mm", "convert", "--tier", "tiny", "--dir", dir, "-o", filepath.Join(dir, "no", "such", "o.json")); err == nil {
		t.Error("unwritable output accepted")
	}
	if _, err := run(t, "mm", "convert", "--tier", "nope"); err == nil {
		t.Error("unknown tier accepted")
	}
}

// FuzzMMSafeName: whatever the name, a name that passes is a plain
// [A-Z0-9_-] string of at most 32 bytes, never a path or a shell word.
func FuzzMMSafeName(f *testing.F) {
	for _, s := range []string{"HS21", "CVXQP1_S", "", "../x", "a b", "HS35MOD"} {
		f.Add(s)
	}
	f.Fuzz(func(t *testing.T, s string) {
		if !mmSafeName(s) {
			return
		}
		if len(s) > 32 || strings.ContainsAny(s, "/\\ ;$`\x00.") || filepath.Base(s) != s {
			t.Fatalf("%q passed", s)
		}
	})
}

func TestMMCommandMount(t *testing.T) {
	argv, d := mmCommand("", "/cache/mm")
	if d != "/cache/mm" || argv[0] != "uv" {
		t.Errorf("host: %v %s", argv, d)
	}
	argv, d = mmCommand("linux/amd64", "/cache/mm")
	if d != "/mm" || strings.Join(argv[:7], " ") != "docker run --rm -i -v /cache/mm:/mm:ro --platform" {
		t.Errorf("container: %v %s", argv[:7], d)
	}
	want := command("linux/amd64")
	if len(argv) != len(want)+2 || argv[len(argv)-1] != want[len(want)-1] {
		t.Errorf("the container command changed beyond the mount: %v", argv)
	}
}
