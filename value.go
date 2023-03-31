package main

import (
	"math"
)

var counter int

// Value defines an individual value in micrograd-go. This holds a specified floating point number aswell as
// the previous values which were used to calculate the current value. This struct also holds the gradient of
// the value which is determined via the operations used to compute it. This is useful to determine the impact
// that changing this value will have and scales to support a large number of operations.
type Value struct {
	id   int
	data float64
	grad float64

	prev      []*Value
	_backward func()
}

// TODO: Documentation
func New(data float64, prev []*Value) *Value {
	out := Value{
		data:      data,
		grad:      0,
		id:        counter,
		prev:      prev,
		_backward: func() {},
	}
	counter++
	return &out
}

func NewArray(data []float64) []*Value {
	array := make([]*Value, len(data))

	for i, n := range data {
		out := Value{
			data:      n,
			grad:      0,
			id:        counter,
			prev:      make([]*Value, 0),
			_backward: func() {},
		}
		array[i] = &out
		counter++
	}
	return array
}

func NewConstant(data float64) *Value {
	out := Value{
		data:      data,
		grad:      0,
		id:        counter,
		prev:      make([]*Value, 0),
		_backward: func() {},
	}
	counter++
	return &out
}

func (self *Value) Add(other *Value) *Value {
	prev := []*Value{self, other}
	out := New(self.data+other.data, prev)

	out._backward = func() {
		// f(a) = a + b, f'(a) = 1
		self.grad += out.grad
		other.grad += out.grad
	}
	return out
}

func (self *Value) Mul(other *Value) *Value {
	prev := []*Value{self, other}
	out := New(self.data*other.data, prev)

	out._backward = func() {
		// f(a) = ab, f'(a) = b
		self.grad += other.data * out.grad
		other.grad += self.data * out.grad
	}
	return out
}

func (self *Value) Pow(other float64) *Value {
	prev := []*Value{self}
	out := New(math.Pow(self.data, other), prev)

	out._backward = func() {
		// f(a) = a**c, f'(a) = c*a**(c-1)
		self.grad += (other * math.Pow(self.data, other-1)) * out.grad
	}
	return out
}

func (self *Value) Tanh() *Value {
	prev := []*Value{self}
	out := New(math.Tanh(self.data), prev)

	out._backward = func() {
		// f(a) = tanh(a), f'(a) = 1-(tanh(a))**2
		self.grad += (1 - math.Pow(math.Tanh(self.data), 2)) * out.grad
	}
	return out
}

func (self *Value) Div(other *Value) *Value {
	return self.Mul(other.Pow(-1))
}

func (root *Value) Backward() {
	root.grad = 1.0
	topo := NewTopologicalSort()
	topo.Sort(root)
	list := topo.sorted

	for i := range list {
		list[len(list)-1-i]._backward()
	}
}
