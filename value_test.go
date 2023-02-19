package main

import (
	"fmt"
	"math"
	"testing"
)

const (
	h = 0.0001
)

func f(a, b, c float64) float64 {
	return (a + b) * c
}

func y(a, b, c float64) float64 {
	d := math.Pow(a, 3.5)
	e := d * b
	f := math.Pow(e, 2)
	g := f / c
	return g
}

func checkGradient(ana, num float64) error {
	if math.Abs((ana-num)/ana)*100 > 0.1 {
		return fmt.Errorf("gradient mismatch, analytical %f, numerical %f", ana, num)
	}
	return nil
}

func TestBasic(t *testing.T) {
	a := NewConstant(10)
	b := NewConstant(-20)
	c := NewConstant(30)
	d := a.Add(b)
	e := d.Mul(c)
	e.grad = 1.0
	c._backward()
	e._backward()
	d._backward()

	// Compute answers numerically using first principles.
	a_ans := (f(10+h, -20, 30) - f(10, -20, 30)) / h
	b_ans := (f(10, -20+h, 30) - f(10, -20, 30)) / h
	c_ans := (f(10, -20, 30+h) - f(10, -20, 30)) / h

	err := checkGradient(a.grad, a_ans)
	if err != nil {
		t.Errorf("comparisn error, %s", err)
	}

	err = checkGradient(b.grad, b_ans)
	if err != nil {
		t.Errorf("comparisn error, %s", err)
	}

	err = checkGradient(c.grad, c_ans)
	if err != nil {
		t.Errorf("comparisn error, %s", err)
	}
}

func TestDiv(t *testing.T) {
	// (((a**3.5) * b)**2) / c
	a := NewConstant(2)
	b := NewConstant(-4)
	c := NewConstant(6)
	d := a.Pow(3.5)
	e := d.Mul(b)
	f := e.Pow(2)
	g := f.Div(c)
	g.grad = 1.0
	g.Backward()

	// Compute answers numerically using first principles.
	a_ans := (y(2+h, -4, 6) - y(2, -4, 6)) / h
	b_ans := (y(2, -4+h, 6) - y(2, -4, 6)) / h
	c_ans := (y(2, -4, 6+h) - y(2, -4, 6)) / h

	err := checkGradient(a.grad, a_ans)
	if err != nil {
		t.Errorf("comparisn error, %s", err)
	}

	err = checkGradient(b.grad, b_ans)
	if err != nil {
		t.Errorf("comparisn error, %s", err)
	}

	err = checkGradient(c.grad, c_ans)
	if err != nil {
		t.Errorf("comparisn error, %s", err)
	}
}
