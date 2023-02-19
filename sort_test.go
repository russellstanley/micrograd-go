package main

import (
	"fmt"
	"testing"
)

func isEqual(actual []int, result []*Value) error {
	for i := range actual {
		if actual[i] != result[i].id {
			return fmt.Errorf("sorting error")
		}
	}
	return nil
}

func TestSortBasic(t *testing.T) {
	s := counter
	a := NewConstant(1)
	b := NewConstant(2)
	c := a.Add(b)
	d := c.Mul(a)
	e := c.Div(d)

	sorter := NewTopologicalSort()
	sorter.Sort(e)

	ans_visited := []int{s + 5, s + 2, s + 0, s + 1, s + 4, s + 3}
	ans_sorted := []int{s, s + 1, s + 2, s + 3, s + 4, s + 5}

	err := isEqual(ans_visited, sorter.visited)
	if err != nil {
		t.Error(err)
	}

	err = isEqual(ans_sorted, sorter.sorted)
	if err != nil {
		t.Error(err)
	}
}

func TestSort(t *testing.T) {
	s := counter
	a := NewConstant(2)
	b := NewConstant(-4)
	c := NewConstant(6)
	d := a.Pow(3.5)
	e := d.Mul(b)
	f := e.Pow(2)
	g := f.Div(c)

	sorter := NewTopologicalSort()
	sorter.Sort(g)

	ans_visited := []int{s + 7, s + 5, s + 4, s + 3, s, s + 1, s + 6, s + 2}
	ans_sorted := []int{s, s + 3, s + 1, s + 4, s + 5, s + 2, s + 6, s + 7}

	err := isEqual(ans_visited, sorter.visited)
	if err != nil {
		for i := range ans_visited {
			t.Logf("ans: %d, got %d\n", ans_visited[i], sorter.visited[i].id)
		}
		t.Error(err)
	}

	err = isEqual(ans_sorted, sorter.sorted)
	if err != nil {
		for i := range ans_visited {
			t.Logf("ans: %d, got %d\n", ans_visited[i], sorter.visited[i].id)
		}
		t.Error(err)
	}
}
