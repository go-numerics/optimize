package optimize

import "math"

// Subroutines of L-BFGS-B 3.0, one method per reference subroutine.

// active projects x onto the feasible set and initialises iwhere and the
// cnstnd and boxed flags (reference subroutine active).
func (s *lbfgsb) active(x []float64) {
	l, u, nbd := s.l, s.u, s.nbd
	for i := range x {
		if nbd[i] > nbdFree {
			if nbd[i] <= nbdBoth && x[i] <= l[i] {
				x[i] = l[i]
			} else if nbd[i] >= nbdBoth && x[i] >= u[i] {
				x[i] = u[i]
			}
		}
	}
	s.boxed = true
	for i := range x {
		if nbd[i] != nbdBoth {
			s.boxed = false
		}
		if nbd[i] == nbdFree {
			s.iwhere[i] = -1
		} else {
			s.cnstnd = true
			if nbd[i] == nbdBoth && u[i]-l[i] <= 0 {
				s.iwhere[i] = 3
			} else {
				s.iwhere[i] = 0
			}
		}
	}
}

// projgr returns the infinity norm of the projected gradient (reference
// subroutine projgr). A NaN gradient component is returned as is.
func (s *lbfgsb) projgr(x, g []float64) float64 {
	sbgnrm := 0.0
	for i, gi := range g {
		if math.IsNaN(gi) {
			return gi
		}
		if s.nbd[i] != nbdFree {
			if gi < 0 {
				if s.nbd[i] >= nbdBoth {
					gi = cfmax(x[i]-s.u[i], gi)
				}
			} else if s.nbd[i] <= nbdBoth {
				gi = cfmin(x[i]-s.l[i], gi)
			}
		}
		sbgnrm = cfmax(sbgnrm, math.Abs(gi))
	}
	return sbgnrm
}

// bmv computes p = M·v for the 2col×2col middle matrix
//
//	M = [ -D    L'  ]⁻¹
//	    [  L   θS'S ]
//
// using the Cholesky factor of θS'S + LD⁻¹L' held in wt (reference
// subroutine bmv). The factor comes from a successful dpofa, so its diagonal
// is positive and the triangular solves cannot fail. Callers ensure col > 0
// (the reference code returns early otherwise).
func (s *lbfgsb) bmv(v, p []float64) {
	m, col, sy := s.m, s.col, s.sy
	// PART I: solve [ D^(1/2)      O ] [ p1 ] = [ v1 ]
	//               [ -L*D^(-1/2)  J ] [ p2 ]   [ v2 ].
	p[col] = v[col]
	for i := 1; i < col; i++ {
		i2 := col + i
		sum := 0.0
		for k := 0; k <= i-1; k++ {
			sum += sy[i+m*k] * v[k] / sy[k+m*k]
		}
		p[i2] = v[i2] + sum
	}
	dtrsl(s.wt, m, col, p[col:], true)
	for i := 0; i < col; i++ {
		p[i] = v[i] / math.Sqrt(sy[i+m*i])
	}
	// PART II: solve [ -D^(1/2)   D^(-1/2)*L'  ] [ p1 ] = [ p1 ]
	//                [  0         J'           ] [ p2 ]   [ p2 ].
	dtrsl(s.wt, m, col, p[col:], false)
	for i := 0; i < col; i++ {
		p[i] = -p[i] / math.Sqrt(sy[i+m*i])
	}
	for i := 0; i < col; i++ {
		sum := 0.0
		for k := i + 1; k < col; k++ {
			sum += sy[k+m*i] * p[col+k] / sy[i+m*i]
		}
		p[i] += sum
	}
}

// cauchy computes the generalized Cauchy point into z: the first local
// minimiser of the quadratic model along the projected steepest-descent
// path, examining the breakpoints in increasing order with a heap (reference
// subroutine cauchy). On return wa[2m:4m] holds c = W'(z - x).
func (s *lbfgsb) cauchy(x, g []float64) {
	n, m, col := s.n, s.m, s.col
	l, u, nbd, iwhere := s.l, s.u, s.nbd, s.iwhere
	t, d, xcp, iorder := s.t, s.d, s.z, s.indx2
	wy, ws, theta := s.wy, s.ws, s.theta
	p, c, wbp, v := s.wa[0:2*m], s.wa[2*m:4*m], s.wa[4*m:6*m], s.wa[6*m:8*m]

	if s.sbgnrm <= 0 {
		copy(xcp, x)
		return
	}
	bnded := true
	nfree := n
	nbreak := -1
	ibkmin := 0
	bkmin := 0.0
	col2 := 2 * col
	f1, tl, tu := 0.0, 0.0, 0.0
	for i := 0; i < col2; i++ {
		p[i] = 0
	}
	// Set the break points and the direction d = -g projected.
	for i := 0; i < n; i++ {
		neggi := -g[i]
		if iwhere[i] != 3 && iwhere[i] != -1 {
			if nbd[i] <= nbdBoth {
				tl = x[i] - l[i]
			}
			if nbd[i] >= nbdBoth {
				tu = u[i] - x[i]
			}
			xlower := nbd[i] <= nbdBoth && tl <= 0
			xupper := nbd[i] >= nbdBoth && tu <= 0
			iwhere[i] = 0
			if xlower {
				if neggi <= 0 {
					iwhere[i] = 1
				}
			} else if xupper {
				if neggi >= 0 {
					iwhere[i] = 2
				}
			} else if math.Abs(neggi) <= 0 {
				iwhere[i] = -3
			}
		}
		pointr := s.head
		if iwhere[i] != 0 && iwhere[i] != -1 {
			d[i] = 0
			continue
		}
		d[i] = neggi
		f1 -= float64(neggi * neggi)
		for j := 0; j < col; j++ {
			p[j] += float64(wy[i+n*pointr] * neggi)
			p[col+j] += float64(ws[i+n*pointr] * neggi)
			pointr = (pointr + 1) % m
		}
		switch {
		case nbd[i] <= nbdBoth && nbd[i] != nbdFree && neggi < 0:
			nbreak++
			iorder[nbreak] = i
			t[nbreak] = tl / (-neggi)
			if nbreak == 0 || t[nbreak] < bkmin {
				bkmin = t[nbreak]
				ibkmin = nbreak
			}
		case nbd[i] >= nbdBoth && neggi > 0:
			nbreak++
			iorder[nbreak] = i
			t[nbreak] = tu / neggi
			if nbreak == 0 || t[nbreak] < bkmin {
				bkmin = t[nbreak]
				ibkmin = nbreak
			}
		default:
			nfree--
			iorder[nfree] = i
			if math.Abs(neggi) > 0 {
				bnded = false
			}
		}
	}
	if theta != 1 {
		for j := col; j < col2; j++ {
			p[j] *= theta
		}
	}
	copy(xcp, x)
	// The reference code returns here when nbreak == -1 and nfree == n (d is
	// zero). That needs every variable skipped above, i.e. a zero projected
	// gradient, which the sbgnrm <= 0 test has already handled.
	for j := 0; j < col2; j++ {
		c[j] = 0
	}
	// Initial f2 = -θf1 - p'Mp.
	f2 := float64(-theta * f1)
	f2org := f2
	if col > 0 {
		s.bmv(p, v)
		f2 -= dot(v[:col2], p[:col2])
	}
	dtm := -f1 / f2
	tsum := 0.0
	nleft := nbreak
	iter := 0
	tj := 0.0
	// Search the first local minimiser along the piecewise linear path.
	for nbreak > -1 {
		tj0 := tj
		var ibp int
		if iter == 0 {
			tj = bkmin
			ibp = iorder[ibkmin]
		} else {
			if iter == 1 && ibkmin != nbreak {
				// Replace the already used smallest breakpoint with the
				// last one, then build a heap.
				t[ibkmin] = t[nbreak]
				iorder[ibkmin] = iorder[nbreak]
			}
			hpsolb(nleft, t, iorder, iter-1)
			tj = t[nleft]
			ibp = iorder[nleft]
		}
		dt := tj - tj0
		if dtm < dt {
			// The minimiser lies within this interval.
			if dtm <= 0 {
				dtm = 0
			}
			tsum += dtm
			daxpy(tsum, d, xcp)
			if col > 0 {
				daxpy(dtm, p[:col2], c[:col2])
			}
			return
		}
		// Otherwise fix one variable at its bound and update f1, f2.
		tsum += dt
		nleft--
		iter++
		dibp := d[ibp]
		d[ibp] = 0
		var zibp float64
		if dibp > 0 {
			zibp = u[ibp] - x[ibp]
			xcp[ibp] = u[ibp]
			iwhere[ibp] = 2
		} else {
			zibp = l[ibp] - x[ibp]
			xcp[ibp] = l[ibp]
			iwhere[ibp] = 1
		}
		if nleft == -1 && nbreak == n-1 {
			// All n variables are fixed: the Cauchy point is xcp.
			dtm = dt
			if col > 0 {
				daxpy(dtm, p[:col2], c[:col2])
			}
			return
		}
		dibp2 := float64(dibp * dibp)
		f1 = f1 + float64(dt*f2) + dibp2 - float64(theta*dibp*zibp)
		f2 -= float64(theta * dibp2)
		if col > 0 {
			daxpy(dt, p[:col2], c[:col2])
			// wbp = row of W corresponding to the breakpoint.
			pointr := s.head
			for j := 0; j < col; j++ {
				wbp[j] = wy[ibp+n*pointr]
				wbp[col+j] = float64(theta * ws[ibp+n*pointr])
				pointr = (pointr + 1) % m
			}
			s.bmv(wbp, v)
			wmc := dot(c[:col2], v[:col2])
			wmp := dot(p[:col2], v[:col2])
			wmw := dot(wbp[:col2], v[:col2])
			daxpy(-dibp, wbp[:col2], p[:col2])
			f1 += float64(dibp * wmc)
			f2 = f2 + float64(dibp*2*wmp) - float64(dibp2*wmw)
		}
		f2 = cfmax(epsmch*f2org, f2)
		if nleft >= 0 {
			dtm = -f1 / f2
			continue
		}
		if bnded {
			dtm = 0
		} else {
			dtm = -f1 / f2
		}
		break
	}
	if dtm <= 0 {
		dtm = 0
	}
	tsum += dtm
	daxpy(tsum, d, xcp)
	if col > 0 {
		daxpy(dtm, p[:col2], c[:col2])
	}
}

// daxpy computes y += a·x.
func daxpy(a float64, x, y []float64) {
	for i := range x {
		y[i] += float64(a * x[i])
	}
}

// hpsolb manages the heap of the breakpoints t[0..n] with their indices
// iorder (reference subroutine hpsolb). With iheap == 0 it first builds the
// heap; then, if n > 0, it moves the least element to t[n] and restores the
// heap on t[0..n-1].
func hpsolb(n int, t []float64, iorder []int, iheap int) {
	if iheap == 0 {
		// Rearrange t[0..n] into a heap (1-based positions k).
		for k := 2; k <= n+1; k++ {
			ddum := t[k-1]
			indxin := iorder[k-1]
			i := k
			for i > 1 {
				j := i / 2
				if !(ddum < t[j-1]) {
					break
				}
				t[i-1] = t[j-1]
				iorder[i-1] = iorder[j-1]
				i = j
			}
			t[i-1] = ddum
			iorder[i-1] = indxin
		}
	}
	if n > 0 {
		// Remove the least element and re-sift.
		i := 1
		out := t[0]
		indxout := iorder[0]
		ddum := t[n]
		indxin := iorder[n]
		for {
			j := i + i
			if j > n {
				break
			}
			if t[j] < t[j-1] {
				j++
			}
			if !(t[j-1] < ddum) {
				break
			}
			t[i-1] = t[j-1]
			iorder[i-1] = iorder[j-1]
			i = j
		}
		t[i-1] = ddum
		iorder[i-1] = indxin
		t[n] = out
		iorder[n] = indxout
	}
}

// freev partitions the variables into free (index[0:nfree]) and active ones
// and, after the first iteration of a constrained problem, records the
// variables entering (indx2[0:nenter]) and leaving (indx2[ileave:n]) the
// free set (reference subroutine freev).
func (s *lbfgsb) freev() {
	n, idx, idx2, iwhere := s.n, s.index, s.indx2, s.iwhere
	s.nenter = 0
	s.ileave = n
	if s.iter > 0 && s.cnstnd {
		for i := 0; i < s.nfree; i++ {
			k := idx[i]
			if iwhere[k] > 0 {
				s.ileave--
				idx2[s.ileave] = k
			}
		}
		for i := s.nfree; i < n; i++ {
			k := idx[i]
			if iwhere[k] <= 0 {
				idx2[s.nenter] = k
				s.nenter++
			}
		}
	}
	s.nfree = 0
	iact := n
	for i := 0; i < n; i++ {
		if iwhere[i] <= 0 {
			idx[s.nfree] = i
			s.nfree++
		} else {
			iact--
			idx[iact] = i
		}
	}
}

// cmprlb computes the reduced gradient r = -Z'B(z - x) - Z'g of the
// quadratic model at the Cauchy point, restricted to the free variables
// (reference subroutine cmprlb). It uses c from cauchy in wa[2m:4m].
func (s *lbfgsb) cmprlb(x, g []float64) {
	n, m, col, theta := s.n, s.m, s.col, s.theta
	r, z, index, wa := s.r, s.z, s.index, s.wa
	if !s.cnstnd && col > 0 {
		for i := 0; i < n; i++ {
			r[i] = -g[i]
		}
		return
	}
	for i := 0; i < s.nfree; i++ {
		k := index[i]
		r[i] = float64(-theta*(z[k]-x[k])) - g[k]
	}
	s.bmv(wa[2*m:4*m], wa[0:2*m])
	pointr := s.head
	for j := 0; j < col; j++ {
		a1 := wa[j]
		a2 := float64(theta * wa[col+j])
		for i := 0; i < s.nfree; i++ {
			k := index[i]
			r[i] = r[i] + float64(s.wy[k+n*pointr]*a1) + float64(s.ws[k+n*pointr]*a2)
		}
		pointr = (pointr + 1) % m
	}
}

// matupd stores the new correction pair (d, r) and updates S'Y and S'S
// (reference subroutine matupd).
func (s *lbfgsb) matupd(rr, dr float64) {
	n, m := s.n, s.m
	ws, wy, sy, ss, d, r := s.ws, s.wy, s.sy, s.ss, s.d, s.r
	if s.iupdat <= m {
		s.col = s.iupdat
		s.itail = (s.head + s.iupdat - 1) % m
	} else {
		s.itail = (s.itail + 1) % m
		s.head = (s.head + 1) % m
	}
	copy(ws[s.itail*n:(s.itail+1)*n], d)
	copy(wy[s.itail*n:(s.itail+1)*n], r)
	s.theta = rr / dr
	col := s.col
	if s.iupdat > m {
		// Move the old information: drop the oldest pair.
		for j := 1; j < col; j++ {
			copy(ss[m*(j-1):m*(j-1)+j], ss[1+m*j:1+m*j+j])
			for k := 0; k < col-j; k++ {
				sy[(j-1)+k+m*(j-1)] = sy[j+k+m*j]
			}
		}
	}
	// Add the new information: the last row of SY and the last column of SS.
	pointr := s.head
	for j := 0; j < col-1; j++ {
		sy[col-1+m*j] = dot(d, wy[pointr*n:(pointr+1)*n])
		ss[j+m*(col-1)] = dot(ws[pointr*n:(pointr+1)*n], d)
		pointr = (pointr + 1) % m
	}
	if s.stp == 1 {
		ss[(col-1)+m*(col-1)] = s.dtd
	} else {
		ss[(col-1)+m*(col-1)] = float64(s.stp * s.stp * s.dtd)
	}
	sy[(col-1)+m*(col-1)] = dr
}

// formt forms the upper half of T = θS'S + LD⁻¹L' in wt and overwrites it
// with its Cholesky factor (reference subroutine formt). It reports false
// when T is not positive definite.
func (s *lbfgsb) formt() bool {
	m, col, theta := s.m, s.col, s.theta
	wt, sy, ss := s.wt, s.sy, s.ss
	for j := 0; j < col; j++ {
		wt[m*j] = float64(theta * ss[m*j])
	}
	for i := 1; i < col; i++ {
		for j := i; j < col; j++ {
			k1 := min(i, j)
			ddum := 0.0
			for k := 0; k < k1; k++ {
				ddum += sy[i+m*k] * sy[j+m*k] / sy[k+m*k]
			}
			wt[i+m*j] = ddum + float64(theta*ss[i+m*j])
		}
	}
	return dpofa(wt, m, col)
}

// formk forms the LEL' factorisation of the 2col×2col indefinite matrix
//
//	K = [-D - Y'ZZ'Y/θ       L_a' - R_z' ]
//	    [ L_a - R_z    θS'AA'S          ]
//
// used by the subspace minimisation, updating its parts in snd with the
// variables that entered or left the free set (reference subroutine formk).
// It reports false when a Cholesky factorisation fails.
func (s *lbfgsb) formk() bool {
	n, m, col, head, theta := s.n, s.m, s.col, s.head, s.theta
	ws, wy, sy, wn, wn1 := s.ws, s.wy, s.sy, s.wn, s.snd
	ind, indx2 := s.index, s.indx2
	nsub, nenter, ileave := s.nfree, s.nenter, s.ileave
	m2 := 2 * m

	// Form the lower triangle of WN1 = [Y'ZZ'Y   L_a'+R_z']
	//                                  [L_a+R_z   S'AA'S   ].
	var upcl int
	if s.updatd {
		if s.iupdat > m {
			// Shift the old part of WN1.
			for jy := 0; jy < m-1; jy++ {
				js := m + jy
				copy(wn1[jy+m2*jy:jy+m2*jy+m-(jy+1)], wn1[(jy+1)+m2*(jy+1):(jy+1)+m2*(jy+1)+m-(jy+1)])
				copy(wn1[js+m2*js:js+m2*js+m-(jy+1)], wn1[(js+1)+m2*(js+1):(js+1)+m2*(js+1)+m-(jy+1)])
				copy(wn1[m+m2*jy:m+m2*jy+m-1], wn1[(m+1)+m2*(jy+1):(m+1)+m2*(jy+1)+m-1])
			}
		}
		// Put the new rows in blocks (1,1), (2,1) and (2,2).
		iy := col - 1
		is := m + col - 1
		ipntr := head + col - 1
		if ipntr >= m {
			ipntr -= m
		}
		jpntr := head
		for jy := 0; jy < col; jy++ {
			js := m + jy
			temp1, temp2, temp3 := 0.0, 0.0, 0.0
			for k := 0; k < nsub; k++ {
				k1 := ind[k]
				temp1 += float64(wy[k1+n*ipntr] * wy[k1+n*jpntr])
			}
			for k := nsub; k < n; k++ {
				k1 := ind[k]
				temp2 += float64(ws[k1+n*ipntr] * ws[k1+n*jpntr])
				temp3 += float64(ws[k1+n*ipntr] * wy[k1+n*jpntr])
			}
			wn1[iy+m2*jy] = temp1
			wn1[is+m2*js] = temp2
			wn1[is+m2*jy] = temp3
			jpntr = (jpntr + 1) % m
		}
		// Put the new column in block (2,1).
		jy := col - 1
		jpntr = head + col - 1
		if jpntr >= m {
			jpntr -= m
		}
		ipntr = head
		for i := 0; i < col; i++ {
			is := m + i
			temp3 := 0.0
			for k := 0; k < nsub; k++ {
				k1 := ind[k]
				temp3 += float64(ws[k1+n*ipntr] * wy[k1+n*jpntr])
			}
			ipntr = (ipntr + 1) % m
			wn1[is+m2*jy] = temp3
		}
		upcl = col - 1
	} else {
		upcl = col
	}
	// Modify the old parts in blocks (1,1) and (2,2) for the changes in
	// the free set.
	ipntr := head
	for iy := 0; iy < upcl; iy++ {
		is := m + iy
		jpntr := head
		for jy := 0; jy <= iy; jy++ {
			js := m + jy
			temp1, temp2, temp3, temp4 := 0.0, 0.0, 0.0, 0.0
			for k := 0; k < nenter; k++ {
				k1 := indx2[k]
				temp1 += float64(wy[k1+n*ipntr] * wy[k1+n*jpntr])
				temp2 += float64(ws[k1+n*ipntr] * ws[k1+n*jpntr])
			}
			for k := ileave; k < n; k++ {
				k1 := indx2[k]
				temp3 += float64(wy[k1+n*ipntr] * wy[k1+n*jpntr])
				temp4 += float64(ws[k1+n*ipntr] * ws[k1+n*jpntr])
			}
			wn1[iy+m2*jy] = wn1[iy+m2*jy] + temp1 - temp3
			wn1[is+m2*js] = wn1[is+m2*js] - temp2 + temp4
			jpntr = (jpntr + 1) % m
		}
		ipntr = (ipntr + 1) % m
	}
	// Modify the old parts in block (2,1).
	ipntr = head
	for is := m; is < m+upcl; is++ {
		jpntr := head
		for jy := 0; jy < upcl; jy++ {
			temp1, temp3 := 0.0, 0.0
			for k := 0; k < nenter; k++ {
				k1 := indx2[k]
				temp1 += float64(ws[k1+n*ipntr] * wy[k1+n*jpntr])
			}
			for k := ileave; k < n; k++ {
				k1 := indx2[k]
				temp3 += float64(ws[k1+n*ipntr] * wy[k1+n*jpntr])
			}
			if is <= jy+m {
				wn1[is+m2*jy] = wn1[is+m2*jy] + temp1 - temp3
			} else {
				wn1[is+m2*jy] = wn1[is+m2*jy] - temp1 + temp3
			}
			jpntr = (jpntr + 1) % m
		}
		ipntr = (ipntr + 1) % m
	}
	// Form the upper triangle of WN = [D+Y'ZZ'Y/θ   -L_a'+R_z' ]
	//                                 [-L_a+R_z      S'AA'Sθ   ].
	for iy := 0; iy < col; iy++ {
		is := col + iy
		is1 := m + iy
		for jy := 0; jy <= iy; jy++ {
			js := col + jy
			js1 := m + jy
			wn[jy+m2*iy] = wn1[iy+m2*jy] / theta
			wn[js+m2*is] = float64(wn1[is1+m2*js1] * theta)
		}
		for jy := 0; jy < iy; jy++ {
			wn[jy+m2*is] = -wn1[is1+m2*jy]
		}
		for jy := iy; jy < col; jy++ {
			wn[jy+m2*is] = wn1[is1+m2*jy]
		}
		wn[iy+m2*iy] += sy[iy+m*iy]
	}
	// Form the upper triangle of WN = [  LL'            L⁻¹(-L_a'+R_z')]
	//                                 [(-L_a+R_z)L'⁻¹   S'AA'Sθ        ]:
	// first Cholesky factor (1,1) block. It is D + Y'ZZ'Y/θ with D > 0, so
	// in practice only the (2,2) factorisation fails.
	ok := dpofa(wn, m2, col)
	if ok {
		// Then form L⁻¹(-L_a'+R_z') in the (1,2) block.
		col2 := 2 * col
		for js := col; js < col2; js++ {
			dtrsl(wn, m2, col, wn[m2*js:m2*js+col], true)
		}
		// Form S'AA'Sθ + (L⁻¹(-L_a'+R_z'))'L⁻¹(-L_a'+R_z') in the upper
		// triangle of (2,2).
		for is := col; is < col2; is++ {
			for js := is; js < col2; js++ {
				wn[is+m2*js] += dot(wn[m2*is:m2*is+col], wn[m2*js:m2*js+col])
			}
		}
		// Cholesky factorisation of the (2,2) block.
		ok = dpofa(wn[col+m2*col:], m2, col)
	}
	return ok
}

// subsm performs the subspace minimisation: it computes the Newton point of
// the quadratic model over the free variables starting from the Cauchy point
// z, then (L-BFGS-B 3.0, Morales and Nocedal 2011) projects it onto the box
// and, if the projected point is not a descent direction from x, backtracks
// along the unprojected direction to the feasible region instead (reference
// subroutine subsm).
func (s *lbfgsb) subsm(xx, gg []float64) {
	n, m, col, head, theta := s.n, s.m, s.col, s.head, s.theta
	nsub, ind := s.nfree, s.index
	l, u, nbd := s.l, s.u, s.nbd
	x, d, xp, ws, wy := s.z, s.r, s.xp, s.ws, s.wy
	wv := s.wa
	m2 := 2 * m
	col2 := 2 * col

	// Compute wv = W'Zd.
	pointr := head
	for i := 0; i < col; i++ {
		temp1, temp2 := 0.0, 0.0
		for j := 0; j < nsub; j++ {
			k := ind[j]
			temp1 += float64(wy[k+n*pointr] * d[j])
			temp2 += float64(ws[k+n*pointr] * d[j])
		}
		wv[i] = temp1
		wv[col+i] = float64(theta * temp2)
		pointr = (pointr + 1) % m
	}
	// Compute wv := K⁻¹wv.
	dtrsl(s.wn, m2, col2, wv, true)
	for i := 0; i < col; i++ {
		wv[i] = -wv[i]
	}
	dtrsl(s.wn, m2, col2, wv, false)
	// Compute d = (1/θ)d + (1/θ²)Z'W wv.
	pointr = head
	for jy := 0; jy < col; jy++ {
		js := col + jy
		for i := 0; i < nsub; i++ {
			k := ind[i]
			d[i] = d[i] + wy[k+n*pointr]*wv[jy]/theta + float64(ws[k+n*pointr]*wv[js])
		}
		pointr = (pointr + 1) % m
	}
	temp := 1 / theta
	for i := 0; i < nsub; i++ {
		d[i] *= temp
	}

	// Let us try the projection: d is the Newton direction.
	iword := false
	copy(xp, x)
	for i := 0; i < nsub; i++ {
		k := ind[i]
		dk := d[i]
		xk := x[k]
		switch nbd[k] {
		case nbdFree:
			x[k] = xk + dk
		case nbdLower:
			x[k] = cfmax(l[k], xk+dk)
			if x[k] == l[k] {
				iword = true
			}
		case nbdBoth:
			xk = cfmax(l[k], xk+dk)
			x[k] = cfmin(u[k], xk)
			if x[k] == l[k] || x[k] == u[k] {
				iword = true
			}
		default: // nbdUpper
			x[k] = cfmin(u[k], xk+dk)
			if x[k] == u[k] {
				iword = true
			}
		}
	}
	if !iword {
		return
	}
	// Check the sign of the directional derivative.
	ddp := 0.0
	for i := 0; i < n; i++ {
		ddp += float64((x[i] - xx[i]) * gg[i])
	}
	if ddp <= 0 {
		return
	}
	copy(x, xp)
	// Backtrack along d to the feasible region.
	alpha := 1.0
	temp1 := alpha
	ibd := 0
	for i := 0; i < nsub; i++ {
		k := ind[i]
		dk := d[i]
		if nbd[k] == nbdFree {
			continue
		}
		if dk < 0 && nbd[k] <= nbdBoth {
			temp2 := l[k] - x[k]
			if temp2 >= 0 {
				temp1 = 0
			} else if dk*alpha < temp2 {
				temp1 = temp2 / dk
			}
		} else if dk > 0 && nbd[k] >= nbdBoth {
			temp2 := u[k] - x[k]
			if temp2 <= 0 {
				temp1 = 0
			} else if dk*alpha > temp2 {
				temp1 = temp2 / dk
			}
		}
		if temp1 < alpha {
			alpha = temp1
			ibd = i
		}
	}
	if alpha < 1 {
		dk := d[ibd]
		k := ind[ibd]
		if dk > 0 {
			x[k] = u[k]
			d[ibd] = 0
		} else if dk < 0 {
			x[k] = l[k]
			d[ibd] = 0
		}
	}
	for i := 0; i < nsub; i++ {
		k := ind[i]
		x[k] += float64(alpha * d[i])
	}
}

// lnsrlb performs the line search along d from x with the Moré–Thuente
// search (dcsrch with ftol = 1e-3, gtol = 0.9, xtol = 0.1), evaluating f and
// g through sf (reference subroutine lnsrlb and its reverse-communication
// loop in mainlb). On success x, g and the returned f are at the new point.
// It reports false when d is not a descent direction or maxls steps did not
// satisfy the search; x and g then hold the last trial point, and the start
// point and gradient are in s.t and s.r.
func (s *lbfgsb) lnsrlb(sf *scalarFunc, x, g []float64, f float64, maxls int) (float64, bool) {
	n, l, u, nbd, d := s.n, s.l, s.u, s.nbd, s.d
	s.dnorm = s.dnrm2(d)
	s.dtd = float64(s.dnorm * s.dnorm)
	// Determine the maximum step length.
	s.stpmx = 1e10
	if s.cnstnd {
		if s.iter == 0 {
			s.stpmx = 1
		} else {
			for i := 0; i < n; i++ {
				a1 := d[i]
				if nbd[i] == nbdFree {
					continue
				}
				if a1 < 0 && nbd[i] <= nbdBoth {
					a2 := l[i] - x[i]
					if a2 >= 0 {
						s.stpmx = 0
					} else if a1*s.stpmx < a2 {
						s.stpmx = a2 / a1
					}
				} else if a1 > 0 && nbd[i] >= nbdBoth {
					a2 := u[i] - x[i]
					if a2 <= 0 {
						s.stpmx = 0
					} else if a1*s.stpmx > a2 {
						s.stpmx = a2 / a1
					}
				}
			}
		}
	}
	if s.iter == 0 && !s.boxed {
		s.stp = cfmin(1/s.dnorm, s.stpmx)
	} else {
		s.stp = 1
	}
	copy(s.t, x)
	copy(s.r, g)
	ifun := 0
	s.ls = dcsrch{ftol: 1e-3, gtol: 0.9, xtol: 0.1, stpmin: 0, stpmax: s.stpmx}
	task := lsStart
	for {
		s.gd = dot(g, d)
		if ifun == 0 {
			s.gdold = s.gd
			if s.gd >= 0 {
				// The directional derivative is not negative: d is not a
				// descent direction (info = -4).
				return f, false
			}
		}
		s.stp, task = s.ls.iterate(s.stp, f, s.gd, task)
		if task == lsConv || task == lsWarn {
			return f, true
		}
		ifun++
		if ifun-1 >= maxls {
			return f, false
		}
		if s.stp == 1 {
			copy(x, s.z)
		} else {
			for i := 0; i < n; i++ {
				x[i] = float64(s.stp*d[i]) + s.t[i]
				if nbd[i] == nbdLower || nbd[i] == nbdBoth {
					x[i] = cfmax(x[i], l[i])
				}
				if nbd[i] == nbdBoth || nbd[i] == nbdUpper {
					x[i] = cfmin(x[i], u[i])
				}
			}
		}
		f = sf.funGrad(x, g)
	}
}

// dnrm2 returns the Euclidean norm of v.
func (s *lbfgsb) dnrm2(v []float64) float64 {
	return math.Sqrt(dot(v, v))
}

// dpofa factors the n×n symmetric positive definite matrix whose upper
// triangle is stored column-major in a with leading dimension lda, as R'R
// with R upper triangular overwriting that triangle (LINPACK dpofa). It
// reports false when the matrix is not positive definite.
func dpofa(a []float64, lda, n int) bool {
	for j := 0; j < n; j++ {
		sum := 0.0
		for k := 0; k < j; k++ {
			t := a[k+lda*j] - dot(a[lda*k:lda*k+k], a[lda*j:lda*j+k])
			t /= a[k+lda*k]
			a[k+lda*j] = t
			sum += float64(t * t)
		}
		sum = a[j+lda*j] - sum
		if sum <= 0 {
			return false
		}
		a[j+lda*j] = math.Sqrt(sum)
	}
	return true
}

// dtrsl solves R'x = b (trans) or Rx = b for the n×n upper triangular R
// stored column-major in t with leading dimension ldt, overwriting b
// (LINPACK dtrsl with job 01 and 11). R comes from a successful dpofa, so
// its diagonal is positive.
func dtrsl(t []float64, ldt, n int, b []float64, trans bool) {
	if trans {
		b[0] /= t[0]
		for j := 1; j < n; j++ {
			b[j] = (b[j] - dot(t[ldt*j:ldt*j+j], b[:j])) / t[j+ldt*j]
		}
		return
	}
	b[n-1] /= t[(n-1)+ldt*(n-1)]
	for j := n - 2; j >= 0; j-- {
		daxpy(-b[j+1], t[ldt*(j+1):ldt*(j+1)+j+1], b[:j+1])
		b[j] /= t[j+ldt*j]
	}
}

// cfmax and cfmin are C's fmax and fmin, which the reference code uses: a
// NaN argument is ignored in favour of the other one (math.Max and math.Min
// propagate it).
func cfmax(a, b float64) float64 {
	if math.IsNaN(a) || b > a {
		return b
	}
	return a
}

func cfmin(a, b float64) float64 {
	if math.IsNaN(a) || b < a {
		return b
	}
	return a
}
