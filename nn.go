package main

import (
	"fmt"
	"math/rand"
	"time"
)

// Neuron is a single entity which responds to inputs based on the weight and bias values of the nueron.
type Neuron struct {
	w    []*Value
	bias *Value
}

// NewNeuron will create a new neuron for a specified number of inputs. Weights are intilized randomly.
func NewNeuron(nin int) *Neuron {
	w := make([]*Value, nin)
	for i := range w {
		w[i] = NewConstant(rand.Float64()*2 - 1)
	}

	bias := NewConstant(rand.Float64()*2 - 1)
	return &Neuron{w: w, bias: bias}
}

// fire will compute and return the value of the neuron by multiplying the weights by the inputs. Returns an
// error if the number of weights does not match the inputs.
func (n *Neuron) fire(x []*Value) (*Value, error) {
	if len(x) != len(n.w) {
		return nil, fmt.Errorf("invalid dimensions, input: %d, expected %d", len(x), len(n.w))
	}

	out := NewConstant(0.0)

	for i := range n.w {
		value := n.w[i].Mul(x[i])
		out = out.Add(value)
	}

	// Add a bias node.
	out = out.Add(n.bias)

	// Add the activation function. Currenlty only tanh() supported.
	out = out.Tanh()

	return out, nil
}

// Layer is a collection of neurons.
type Layer struct {
	neurons []*Neuron
}

// NewLayer will create a new layer of neurons based on the specified number of inputs and output.
func NewLayer(nin, nout int) *Layer {
	neurons := make([]*Neuron, nout)

	for i := range neurons {
		neurons[i] = NewNeuron(nin)
	}

	return &Layer{neurons: neurons}
}

// fire will compute and return the output for each neuron in the layer.
func (l *Layer) fire(x []*Value) ([]*Value, error) {
	out := make([]*Value, len(l.neurons))

	for i, n := range l.neurons {
		value, err := n.fire(x)
		if err != nil {
			return nil, fmt.Errorf("layer fire error: %s", err.Error())
		}
		out[i] = value
	}
	return out, nil
}

// MLP defines a multi-layer perception. Containing different layer for a neural network.
type MLP struct {
	layers []*Layer
}

// NewMLP will generate a new MLP.
func NewMLP(nin int, nout []int) *MLP {
	rand.Seed(time.Now().UnixNano())
	nLayer := []int{nin}
	nLayer = append(nLayer, nout...)
	layers := make([]*Layer, len(nout))

	for i := 0; i < len(nout); i++ {
		layers[i] = NewLayer(nLayer[i], nLayer[i+1])
	}

	return &MLP{layers: layers}
}

// fire will compute and return the output for each layer in the neural network.
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

// parameters will return a list of each parameter in the MLP.
func (mlp *MLP) parameters() []*Value {
	out := make([]*Value, 0)

	for _, l := range mlp.layers {
		for _, n := range l.neurons {
			out = append(out, n.w...)
			out = append(out, n.bias)
		}
	}
	return out
}

// zeroGrad will reset the gradient of each paramater to zero.
func (mlp *MLP) zeroGrad(params []*Value) {
	for _, p := range params {
		p.grad = 0.0
	}
}
