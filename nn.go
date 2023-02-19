package main

import (
	"fmt"
	"math/rand"
	"time"
)

type Neuron struct {
	w    []*Value
	bias *Value
}

func NewNeuron(nin int) *Neuron {
	w := make([]*Value, nin)
	for i := range w {
		w[i] = NewConstant(rand.Float64()*2 - 1)
	}

	bias := NewConstant(rand.Float64()*2 - 1)
	return &Neuron{w: w, bias: bias}
}

func (n *Neuron) fire(x []*Value) (*Value, error) {
	if len(x) != len(n.w) {
		return nil, fmt.Errorf("invalid dimensions, input: %d, expected %d", len(x), len(n.w))
	}

	out := NewConstant(0.0)

	for i := range n.w {
		value := n.w[i].Mul(x[i])
		out = out.Add(value)
	}

	out = out.Add(n.bias)
	out = out.Tanh()

	return out, nil
}

type Layer struct {
	nuerons []*Neuron
}

func NewLayer(nin, nout int) *Layer {
	nuerons := make([]*Neuron, nout)

	for i := range nuerons {
		nuerons[i] = NewNeuron(nin)
	}

	return &Layer{nuerons: nuerons}
}

func (l *Layer) fire(x []*Value) ([]*Value, error) {
	out := make([]*Value, len(l.nuerons))

	for i, n := range l.nuerons {
		value, err := n.fire(x)
		if err != nil {
			return nil, fmt.Errorf("layer fire error: %s", err.Error())
		}
		out[i] = value
	}
	return out, nil
}

type MLP struct {
	layers []*Layer
}

func NewMLP(nin int, nout []int) *MLP {
	rand.Seed(time.Now().UnixNano())
	n := []int{nin}
	n = append(n, nout...)
	layers := make([]*Layer, len(nout))

	for i := 0; i < len(nout); i++ {
		layers[i] = NewLayer(n[i], n[i+1])
	}

	return &MLP{layers: layers}
}

func (mlp *MLP) fire(x []*Value) ([]*Value, error) {
	out := x
	var err error
	err = nil

	for _, l := range mlp.layers {
		out, err = l.fire(out)
		if err != nil {
			return nil, err
		}
	}
	return out, nil
}

func (mlp *MLP) parameters() []*Value {
	out := make([]*Value, 0)

	for _, l := range mlp.layers {
		for _, n := range l.nuerons {
			out = append(out, n.w...)
			out = append(out, n.bias)
		}
	}
	return out
}

func (mlp *MLP) forward(xtrain [][]*Value, ytrain []*Value) ([][]*Value, error) {
	ypred := make([][]*Value, len(ytrain))
	var err error

	for i, x := range xtrain {
		ypred[i], err = mlp.fire(x)
		if err != nil {
			return nil, err
		}
	}
	return ypred, nil
}

func (mlp *MLP) zeroGrad() {
	params := mlp.parameters()
	for _, p := range params {
		p.grad = 0.0
	}
}
