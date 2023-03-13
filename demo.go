package main

import (
	"encoding/csv"
	"fmt"
	"os"
	"strconv"
)

func main() {
	mlp := NewMLP(4, []int{16, 6, 3})

	x_train, y_train, err := loadIris("iris.csv")
	if err != nil {
		fmt.Print(err.Error())
		return
	}
	training(mlp, x_train, y_train, 20)
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
				return nil, err
			}
		}

		// Compute the loss.
		l := meanSquared(ytrain, ypred)
		fmt.Println(l.data)

		// Backpropagation.
		l.Backward()
		for _, p := range param {
			p.data += -0.05 * p.grad
		}

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

func loadIris(filePath string) ([][]*Value, []*Value, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, nil, err
	}
	defer file.Close()

	csvReader := csv.NewReader(file)
	data, err := csvReader.ReadAll()
	if err != nil {
		return nil, nil, err
	}

	// Skip header.
	data = data[1:]

	x_train := make([][]*Value, len(data))
	y_train := make([]*Value, len(data))

	for i, entry := range data {
		// Load the labels.
		if entry[4] == "Setosa" {
			y_train[i] = NewConstant(0.0)
		} else if entry[4] == "Versicolor" {
			y_train[i] = NewConstant(1.0)
		} else if entry[4] == "Virginica" {
			y_train[i] = NewConstant(2.0)
		}

		// Load the input data.
		x := make([]float64, 4)

		for j := 0; j < 4; j++ {
			value, err := strconv.ParseFloat(entry[j], 64)
			if err != nil {
				return nil, nil, err
			}
			x[j] = value
		}
		x_train[i] = NewArray(x)
	}
	return x_train, y_train, nil
}

//TODO: Add some visualization.
