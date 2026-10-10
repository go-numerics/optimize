// nofma wraps every float64 multiplication whose result feeds an addition or
// subtraction in an explicit float64() conversion, which the Go spec defines
// as a rounding point that forbids fusing it into a multiply-add.
// Usage: cd tools/nofma && GOWORK=off go run . ../.. file.go...
// It is a separate module so that the library keeps no dependency on x/tools.
package main

import (
	"bytes"
	"fmt"
	"go/ast"
	"go/format"
	"go/importer"
	"go/parser"
	"go/token"
	"go/types"
	"os"
	"path/filepath"
	"strings"

	"golang.org/x/tools/go/ast/astutil"
)

func main() {
	dir := os.Args[1]
	want := map[string]bool{}
	for _, f := range os.Args[2:] {
		want[f] = true
	}
	fset := token.NewFileSet()
	pkgs, err := parser.ParseDir(fset, dir, func(fi os.FileInfo) bool { return !strings.HasSuffix(fi.Name(), "_external_test.go") }, parser.ParseComments)
	if err != nil {
		panic(err)
	}
	for _, pkg := range pkgs {
		var files []*ast.File
		for _, f := range pkg.Files {
			files = append(files, f)
		}
		info := &types.Info{Types: map[ast.Expr]types.TypeAndValue{}}
		conf := types.Config{Importer: importer.ForCompiler(fset, "source", nil), Error: func(error) {}}
		conf.Check(pkg.Name, fset, files, info)
		for name, f := range pkg.Files {
			if !want[filepath.Base(name)] {
				continue
			}
			n := 0
			isFloat := func(e ast.Expr) bool {
				tv, ok := info.Types[e]
				if !ok || tv.Value != nil { // constants never fuse
					return false
				}
				b, ok := tv.Type.Underlying().(*types.Basic)
				return ok && b.Kind() == types.Float64
			}
			isMul := func(e ast.Expr) bool {
				for {
					p, ok := e.(*ast.ParenExpr)
					if !ok {
						break
					}
					e = p.X
				}
				b, ok := e.(*ast.BinaryExpr)
				return ok && b.Op == token.MUL && isFloat(b)
			}
			wrap := func(e ast.Expr) ast.Expr {
				n++
				return &ast.CallExpr{Fun: ast.NewIdent("float64"), Args: []ast.Expr{e}}
			}
			astutil.Apply(f, nil, func(c *astutil.Cursor) bool {
				switch x := c.Node().(type) {
				case *ast.BinaryExpr:
					if (x.Op == token.ADD || x.Op == token.SUB) && isFloat(x) {
						if isMul(x.X) {
							x.X = wrap(x.X)
						}
						if isMul(x.Y) {
							x.Y = wrap(x.Y)
						}
					}
				case *ast.AssignStmt:
					// x op= a*b, and v = a*b whose v may feed a later add
					// (the compiler may fuse across statements).
					for i, r := range x.Rhs {
						if isMul(r) {
							x.Rhs[i] = wrap(r)
						}
					}
				case *ast.ValueSpec:
					for i, r := range x.Values {
						if isMul(r) {
							x.Values[i] = wrap(r)
						}
					}
				case *ast.ReturnStmt:
					for i, r := range x.Results {
						if isMul(r) {
							x.Results[i] = wrap(r)
						}
					}
				}
				return true
			})
			var buf bytes.Buffer
			if err := format.Node(&buf, fset, f); err != nil {
				panic(err)
			}
			os.WriteFile(name, buf.Bytes(), 0o644)
			fmt.Println(filepath.Base(name), n)
		}
	}
}
