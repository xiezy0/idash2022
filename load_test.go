package main

import (
	"fmt"
	"testing"
)

func TestLoadData(t *testing.T) {
	t.Run("load1 weight", func(t *testing.T) {
		s1 := LoadWeight("weight/weight1.csv", 1)
		// fmt.Println(len(s1[0]), len(s1[1018]), len(s1[1019]))
		fmt.Println(s1[4])
		// fmt.Println(len(s1[0]))
	})
	t.Run("load2 data", func(t *testing.T) {
		plaintext := LoadData()
		// fmt.Println(len(plaintext[0]), len(plaintext[1018]), len(plaintext[1019]))
		//for _, pla := range plaintext {
		//	fmt.Println(pla)
		//}
		fmt.Println(plaintext[0][198:250])
		fmt.Println(plaintext[1][0:197])
		fmt.Println(plaintext[1019][1782:1980])
	})
	t.Run("load1 label", func(t *testing.T) {
		data := LoadLabel("label/label_1.csv")
		fmt.Println(data)
		fmt.Println(len(data))
	})
	t.Run("load3 intercept", func(t *testing.T) {
		interslice := LoadIntercept(1)
		// fmt.Println(len(interslice[0]))
		fmt.Println(interslice[2][0], interslice[2][197])
	})
	t.Run("load1 weight plain", func(t *testing.T) {
		LoadWeightPlain("weight/weight1.csv")
	})
	t.Run("load1 data plain", func(t *testing.T) {
		data := LoadDataPlain(1)
		fmt.Println(len(data))
		fmt.Println(data)
	})
	t.Run("load1 data plain1", func(t *testing.T) {
		data := LoadDataRewrite()
		fmt.Println(len(data))
		fmt.Println(data)
	})

}
