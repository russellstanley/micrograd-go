package main

import (
	"fmt"
)

func main() {
	mlp := NewMLP(2, []int{8, 8, 4, 1})

	x_train, y_train, err := loadData("data.csv", 2)
	if err != nil {
		fmt.Print(err.Error())
		return
	}

	training(mlp, x_train, y_train, 20)
	visualization(mlp, x_train, y_train)
}

func training(mlp *MLP, xtrain [][]*Value, ytrain []*Value, generation int) ([][]*Value, error) {
	param := mlp.parameters()
	var err error
	var ypred [][]*Value

	for i := 0; i < generation; i++ {
		// Forward pass.
		ypred = make([][]*Value, len(xtrain))

		for i, x := range xtrain {
			ypred[i], err = mlp.fire(x)
			if err != nil {
				panic(err)
			}
		}

		// Compute the loss.
		l := meanSquared(ytrain, ypred)
		fmt.Println(l.data)

		// Backpropagation.
		l.Backward()
		for _, p := range param {
			p.data += -0.1 * p.grad
		}

		fmt.Printf("%d/%d\n", accuracy(mlp, xtrain, ytrain), len(xtrain))
		mlp.zeroGrad(param)
	}
	return ypred, nil
}

func meanSquared(yt []*Value, ypred [][]*Value) *Value {
	loss := NewConstant(1.0)
	n := len(yt)

	for i := range yt {
		loss = loss.Add(yt[i].Add(
			ypred[i][0].Mul(NewConstant(-1.0))).Pow(2))
	}

	loss = loss.Div(NewConstant(float64(n)))
	return loss
}

func predict(mlp *MLP, x []*Value) float64 {
	pred := 0.0

	out, err := mlp.fire(x)
	if err != nil {
		panic(err)
	}

	if out[0].data > 0.0 {
		pred = 1.0
	} else {
		pred = -1.0
	}
	return pred
}

func accuracy(mlp *MLP, xtrain [][]*Value, ytrain []*Value) int {
	correct := 0

	for i, x := range xtrain {
		pred := predict(mlp, x)

		if pred == ytrain[i].data {
			correct++
		}
	}
	return correct
}
