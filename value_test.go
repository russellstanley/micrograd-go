package main

import (
	"fmt"
	"testing"
)

func f(a, b, c float64) float64 {
	return (a * b) * c
}

func TestBasic(t *testing.T) {
	// (a * b) * c
	h := 0.00001
	fmt.Println((f(100+h, -200, 300) - f(100, -200, 300)) / h)

	a := MakeConstant(100)
	b := MakeConstant(-200)
	c := MakeConstant(300)
	d := a.Mul(b)
	e := d.Mul(c)
	e.grad = 1.0
	c._backward()
	e._backward()
	d._backward()

	fmt.Println(d.prev[0])
	fmt.Println(d.prev[1])
	fmt.Println(e.prev[0])
	fmt.Println(e.prev[1])
	fmt.Println(e)
}
