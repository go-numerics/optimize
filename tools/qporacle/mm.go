package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/spf13/cobra"
)

// The Maros–Mészáros problems as MAT files of qpsolvers/maros_meszaros_qpbenchmark
// (Apache-2.0; its data/ directory, loader maros_meszaros.py read at this commit):
// A = [C; I] stacked, l <= Ax <= u, ±1e20 for infinity, P symmetric. The commit is
// pinned and every file is checked against the SHA-256 below, so a moved branch or
// a tampered download cannot change a fixture.
const (
	mmCommit = "afa738d9a66aca7f5ae223034c72f2e9c6866aff"
	mmURL    = "https://raw.githubusercontent.com/qpsolvers/maros_meszaros_qpbenchmark/" + mmCommit + "/data/"
)

// Tiers of docs/design-qp.md 6.2.
var mmTiers = map[string][]string{
	"tiny": {"GENHS28", "HS118", "HS21", "HS268", "HS35", "HS35MOD", "HS51", "HS52", "HS53", "HS76",
		"LOTSCHD", "QPTEST", "S268", "TAME", "ZECEVIC2", "QAFIRO"},
	"small": {"CVXQP1_S", "CVXQP2_S", "CVXQP3_S", "DUALC1", "DUALC2", "DUALC5", "PRIMALC2", "PRIMALC5",
		"QADLITTL", "QPCBLEND", "QRECIPE", "QSC205", "QSCAGR7", "QSHARE1B", "QSHARE2B", "QSCTAP1", "VALUES"},
}

// mmNames returns the sorted problem names of a tier ("all" is both).
func mmNames(tier string) ([]string, error) {
	var out []string
	switch tier {
	case "tiny", "small":
		out = append(out, mmTiers[tier]...)
	case "all":
		out = append(out, mmTiers["tiny"]...)
		out = append(out, mmTiers["small"]...)
	default:
		return nil, fmt.Errorf("unknown tier %q: want tiny, small or all", tier)
	}
	sort.Strings(out)
	return out, nil
}

// mmSafeName reports whether name is a plain problem name (no path, no shell).
func mmSafeName(name string) bool {
	if name == "" || len(name) > 32 {
		return false
	}
	for _, r := range name {
		if !(r >= 'A' && r <= 'Z' || r >= '0' && r <= '9' || r == '_' || r == '-') {
			return false
		}
	}
	return true
}

// mmCacheDir is where the MAT files go: never inside a repository.
func mmCacheDir() (string, error) {
	d, err := os.UserCacheDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(d, "qporacle", "mm"), nil
}

// httpGet fetches a URL; tests replace it.
var httpGet = func(url string) (io.ReadCloser, error) {
	resp, err := http.Get(url) //nolint:gosec // a constant https URL under a pinned commit
	if err != nil {
		return nil, err
	}
	if resp.StatusCode != http.StatusOK {
		resp.Body.Close()
		return nil, fmt.Errorf("GET %s: %s", url, resp.Status)
	}
	return resp.Body, nil
}

// fetchMM downloads the named files into dir (bounded by maxBytes each) and
// verifies the SHA-256 of each against sums. With record set it verifies nothing
// and returns the sums it saw, to be pinned.
func fetchMM(names []string, dir string, maxBytes int64, record bool) (map[string]string, error) {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, err
	}
	seen := map[string]string{}
	for _, nm := range names {
		if !mmSafeName(nm) {
			return nil, fmt.Errorf("unsafe problem name %q", nm)
		}
		body, err := httpGet(mmURL + nm + ".mat")
		if err != nil {
			return nil, err
		}
		data, err := io.ReadAll(io.LimitReader(body, maxBytes+1))
		body.Close()
		if err != nil {
			return nil, err
		}
		if int64(len(data)) > maxBytes {
			return nil, fmt.Errorf("%s: larger than --max-bytes %d", nm, maxBytes)
		}
		sum := sha256.Sum256(data)
		got := hex.EncodeToString(sum[:])
		seen[nm] = got
		if !record {
			if want, ok := mmSums[nm]; !ok || want != got {
				return nil, fmt.Errorf("%s: sha256 %s, pinned %q", nm, got, want)
			}
		}
		if err := os.WriteFile(filepath.Join(dir, nm+".mat"), data, 0o644); err != nil {
			return nil, err
		}
	}
	return seen, nil
}

// mmCommand returns the oracle command for the MAT directory dir and the
// directory as the script sees it: on the host it is dir itself; in a container
// of the given platform dir is mounted read-only at /mm.
func mmCommand(platform, dir string) ([]string, string) {
	argv := command(platform)
	if platform == "" {
		return argv, dir
	}
	// command(platform) is docker run --rm -i --platform P image uv ...: the mount
	// goes right after "-i".
	out := append([]string{}, argv[:4]...)
	out = append(out, "-v", dir+":/mm:ro")
	out = append(out, argv[4:]...)
	return out, "/mm"
}

func mmCmd() *cobra.Command {
	mm := &cobra.Command{Use: "mm", Short: "The Maros–Mészáros problems: fetch the MAT files, convert them into oracle fixtures"}
	var tier, dir, platform string
	var maxBytes int64
	var record bool
	fetch := &cobra.Command{
		Use: "fetch", Short: "Download the MAT files of a tier at the pinned commit into the cache directory",
		SilenceUsage: true, Args: cobra.NoArgs,
		RunE: func(cmd *cobra.Command, _ []string) error {
			names, err := mmNames(tier)
			if err != nil {
				return err
			}
			if dir == "" {
				if dir, err = mmCacheDir(); err != nil {
					return err
				}
			}
			seen, err := fetchMM(names, dir, maxBytes, record)
			if err != nil {
				return err
			}
			if record {
				for _, nm := range names {
					fmt.Fprintf(cmd.OutOrStdout(), "\t%q: %q,\n", nm, seen[nm])
				}
				return nil
			}
			fmt.Fprintf(cmd.OutOrStdout(), "fetched %d files into %s\n", len(names), dir)
			return nil
		},
	}
	fetch.Flags().StringVar(&tier, "tier", "tiny", "tiny, small or all")
	fetch.Flags().StringVar(&dir, "dir", "", "directory to write (default: the user cache directory)")
	fetch.Flags().Int64Var(&maxBytes, "max-bytes", 1<<20, "refuse a file larger than this")
	fetch.Flags().BoolVar(&record, "record", false, "print the SHA-256 lines to pin instead of verifying")

	var out string
	conv := &cobra.Command{
		Use: "convert", Short: "Run a tier through OSQP, Clarabel and quadprog and write the fixture",
		SilenceUsage: true, Args: cobra.NoArgs,
		RunE: func(cmd *cobra.Command, _ []string) error {
			names, err := mmNames(tier)
			if err != nil {
				return err
			}
			if dir == "" {
				if dir, err = mmCacheDir(); err != nil {
					return err
				}
			}
			for _, nm := range names {
				if _, err := os.Stat(filepath.Join(dir, nm+".mat")); err != nil {
					return fmt.Errorf("%s missing: run mm fetch first: %w", nm, err)
				}
			}
			argv, pdir := mmCommand(platform, dir)
			head := fmt.Sprintf("QP_SET = \"mm\"\nMM_DIR = %q\nMM_NAMES = %q\n", pdir, strings.Join(names, ","))
			var buf bytes.Buffer
			in := append([]byte(head), script...)
			if err := runner(argv, bytes.NewReader(in), &buf, cmd.ErrOrStderr()); err != nil {
				return fmt.Errorf("oracle: %w", err)
			}
			f, err := Decode(&buf)
			if err != nil {
				return fmt.Errorf("oracle output: %w", err)
			}
			if err := CheckCore(f); err != nil {
				return err
			}
			var enc bytes.Buffer
			_ = Encode(&enc, f) // cannot fail: f was just decoded
			if err := os.WriteFile(out, enc.Bytes(), 0o644); err != nil {
				return err
			}
			fmt.Fprintf(cmd.OutOrStdout(), "wrote %d cases to %s\n", len(f.Cases), out)
			return nil
		},
	}
	conv.Flags().StringVar(&tier, "tier", "tiny", "tiny, small or all")
	conv.Flags().StringVar(&dir, "dir", "", "directory of the MAT files (default: the user cache directory)")
	conv.Flags().StringVarP(&out, "output", "o", "testdata/qp/mm/tiny.json", "fixture file to write")
	conv.Flags().StringVar(&platform, "platform", "", "run in a container of this platform (e.g. linux/amd64), the MAT directory mounted read-only")
	mm.AddCommand(fetch, conv)
	return mm
}
