package main

import (
	"testing"
)

func loss(yt []*Value, ypred [][]*Value) *Value {
	loss := NewConstant(0.0)

	for i := range yt {
		loss = loss.Add(
			yt[i].Add(
				ypred[i][0].Mul(NewConstant(-1.0))).Pow(2))
	}
	return loss
}

func train(mlp *MLP, xtrain [][]*Value, ytrain []*Value, generation int, t *testing.T) ([][]*Value, error) {
	param := mlp.parameters()
	var err error
	var ypred [][]*Value
	lp := NewConstant(16.0)

	for i := 0; i < generation; i++ {
		// Forward pass.
		ypred = make([][]*Value, len(xtrain))

		for i, x := range xtrain {
			ypred[i], err = mlp.fire(x)
			if err != nil {
				return nil, err
			}
		}

		// Compute the loss.
		l := loss(ytrain, ypred)
		if l.data > lp.data {
			t.Errorf("loss increased, previous %f, current %f", lp.data, l.data)
		}
		t.Log(l.data)

		// Backpropagation.
		l.Backward()
		for _, p := range param {
			p.data += -0.05 * p.grad
		}

		lp = l
		mlp.zeroGrad(param)
	}
	return ypred, nil
}

func TestNN(t *testing.T) {
	mlp := NewMLP(3, []int{4, 4, 1})

	x1 := NewArray([]float64{2.0, 3.0, -1.0})
	x2 := NewArray([]float64{3.0, -1.0, 0.5})
	x3 := NewArray([]float64{0.5, 1.0, 1.0})
	x4 := NewArray([]float64{1.0, 1.0, -1.0})

	xtrain := [][]*Value{x1, x2, x3, x4}
	ytrain := NewArray([]float64{1.0, -1.0, -1.0, 1.0})

	_, err := train(mlp, xtrain, ytrain, 20, t)
	if err != nil {
		t.Error(err)
	}
}

func TestLoss(t *testing.T) {
	y1 := NewArray([]float64{-1.0})
	y2 := NewArray([]float64{-1.0})
	y3 := NewArray([]float64{-1.0})
	y4 := NewArray([]float64{1.0})

	ytrain := NewArray([]float64{1.0, -1.0, -1.0, 1.0})
	ypred := [][]*Value{y1, y2, y3, y4}
	l := loss(ytrain, ypred)

	if l.data != 4.0 {
		t.Errorf("loss error, got %f, expected %f", l.data, 4.0)
	}
}
