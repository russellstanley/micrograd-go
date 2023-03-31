package main

import (
	"encoding/csv"
	"image/color"
	"os"
	"strconv"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"
)

// visualization will generate a scatter plot of the results. Points that are filled have been correctly predicted.
func visualization(mlp *MLP, xtrain [][]*Value, ytrain []*Value) {
	// Create a new plot.
	p := plot.New()
	p.Title.Text = "Trainning Data"

	// Generate the data points using the training data.
	pts := make(plotter.XYs, len(xtrain))
	for i := range pts {
		pts[i].X = xtrain[i][0].data
		pts[i].Y = xtrain[i][1].data
	}

	colorMap := map[float64]color.RGBA{
		1.0:  {R: 255, G: 0, B: 0, A: 255}, // Red color for even values
		-1.0: {R: 0, G: 0, B: 255, A: 255}, // Blue color for odd values
	}

	// Loop through the data points and set the color and fill based on the predicted value.
	for i := range pts {
		s, err := plotter.NewScatter(plotter.XYs{pts[i]})
		if err != nil {
			panic(err)
		}

		if predict(mlp, xtrain[i]) == ytrain[i].data {
			s.GlyphStyle.Shape = draw.CircleGlyph{}
		}

		s.GlyphStyle.Color = colorMap[ytrain[i].data]
		p.Add(s)
	}

	// Save the plot to a PNG file
	if err := p.Save(4*vg.Inch, 4*vg.Inch, "result.png"); err != nil {
		panic(err)
	}
}

// loadData will load the training data from a csv file.
func loadData(filePath string, attributes int) ([][]*Value, []*Value, error) {
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

	x_train := make([][]*Value, len(data))
	y_train := make([]*Value, len(data))

	for i, entry := range data {
		// Load the training data.
		x := make([]float64, 2)

		for j := 0; j < attributes; j++ {
			value, err := strconv.ParseFloat(entry[j], 64)
			if err != nil {
				return nil, nil, err
			}
			x[j] = value
		}
		x_train[i] = NewArray(x)

		// Load the labels.
		y, err := strconv.ParseFloat(entry[attributes], 64)
		if err != nil {
			return nil, nil, err
		}

		// Update the lables to be -1 or 1.
		if y == 0.0 {
			y_train[i] = NewConstant(-1.0)
		} else {
			y_train[i] = NewConstant(1.0)
		}
	}
	return x_train, y_train, nil
}
