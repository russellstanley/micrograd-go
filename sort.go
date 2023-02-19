package main

type TopologicalSort struct {
	sorted  []*Value
	visited []*Value
}

func NewTopologicalSort() *TopologicalSort {
	return &TopologicalSort{make([]*Value, 0), make([]*Value, 0)}
}

func (t *TopologicalSort) Sort(v *Value) {
	if !t.isVisited(t.visited, v) {
		t.visited = append(t.visited, v)
		for _, i := range v.prev {
			t.Sort(i)
		}
		t.sorted = append(t.sorted, v)
	}
}

func (*TopologicalSort) isVisited(visited []*Value, target *Value) bool {
	for _, i := range visited {
		if target.id == i.id {
			return true
		}
	}
	return false
}
