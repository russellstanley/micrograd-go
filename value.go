package main

import "fmt"

type Value struct {
	data float64
	grad float64

	prev      []*Value
	_backward func()
}

func Make(data float64, prev []*Value) *Value {
	out := Value{
		data:      data,
		prev:      prev,
		_backward: func() { return }}

	return &out
}

func MakeConstant(data float64) *Value {
	out := Value{
		data:      data,
		prev:      make([]*Value, 0),
		_backward: func() { return }}

	return &out
}

func (self *Value) Add(other *Value) *Value {
	prev := []*Value{self, other}
	out := Make(self.data+other.data, prev)

	out._backward = func() {
		// f(a) = a + b, f'(a) = 1
		self.grad += out.grad
		other.grad += out.grad
	}

	return out
}

func (self *Value) Mul(other *Value) *Value {
	prev := []*Value{self, other}
	out := Make(self.data*other.data, prev)

	out._backward = func() {
		// f(a) = ab, f'(a) = b
		self.grad += other.data * out.grad
		other.grad += self.data * out.grad
	}

	return out
}

func f(a, b, c float64) float64 {
	return (a * b) * c
}

func main() {
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
