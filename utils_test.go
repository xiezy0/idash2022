package main

import (
	"fmt"
	"testing"
)

func TestUtils(t *testing.T) {
	t.Run("TASK 1 & NO ENCODE", func(t *testing.T) {
		params := ModelParams{1, 1, 1020, 1}
		evaRes := params.Model()
		label := LoadLabel("label/label_1.csv")
		RMSE := CompareRMSE(evaRes, label)
		fmt.Println("RMSE: ", RMSE)
	})
	t.Run("TASK 2 & NO ENCODE", func(t *testing.T) {
		params := ModelParams{1, 2, 1020, 1}
		evaRes := params.Model()
		label := LoadLabel("label/label_2.csv")
		RMSE := CompareRMSE(evaRes, label)
		fmt.Println("RMSE: ", RMSE)
	})
	t.Run("TASK 3 & NO ENCODE", func(t *testing.T) {
		params := ModelParams{1, 3, 1020, 1}
		evaRes := params.Model()
		label := LoadLabel("label/label_3.csv")
		RMSE := CompareRMSE(evaRes, label)
		fmt.Println("RMSE: ", RMSE)
	})
	t.Run("TASK 4 & NO ENCODE", func(t *testing.T) {
		params := ModelParams{1, 4, 1020, 1}
		evaRes := params.Model()
		label := LoadLabel("label/label_4.csv")
		AUC := CompareAUC(evaRes, label)
		fmt.Println("AUC: ", AUC)
	})
	t.Run("TASK 5 & NO ENCODE", func(t *testing.T) {
		params := ModelParams{1, 5, 1020, 1}
		evaRes := params.Model()
		label := LoadLabel("label/label_5.csv")
		AUC := CompareAUC(evaRes, label)
		fmt.Println("AUC: ", AUC)
	})
	t.Run("XGB TASK 1 & NO ENCODE", func(t *testing.T) {
		params := ModelParams{2, 1, 4, 1}
		evaRes := params.Model()
		label := LoadLabel("label/label_1.csv")
		RMSE := CompareRMSE(evaRes, label)
		fmt.Println("RMSE: ", RMSE)
	})
	t.Run("Model Lasso Parallel for task 1 and task 2", func(t *testing.T) {
		params := ModelParallelParams{1, 1, 6, 4, 1}
		evaRes0, evaRes1 := params.ModelParallel()
		label1 := LoadLabel("label/label_1.csv")
		RMSE1 := CompareRMSE(evaRes0, label1)
		fmt.Println("Task 1 RMSE: ", RMSE1)

		label2 := LoadLabel("label/label_2.csv")
		RMSE2 := CompareRMSE(evaRes1, label2)
		fmt.Println("Task 2 RMSE: ", RMSE2)
	})
	t.Run("Model Lasso Parallel for task 4 and task 5", func(t *testing.T) {
		params := ModelParallelParams{1, 4, 5, 4, 1}
		evaRes0, evaRes1 := params.ModelParallel()
		label4 := LoadLabel("label/label_4.csv")
		AUC4 := CompareAUC(evaRes0, label4)
		fmt.Println("Task 4 AUC: ", AUC4)

		label5 := LoadLabel("label/label_5.csv")
		AUC5 := CompareAUC(evaRes1, label5)
		fmt.Println("Task 5 AUC: ", AUC5)
	})
	t.Run("Model Lasso Parallel for task 1 and task 5", func(t *testing.T) {
		params := ModelParallelParams{1, 1, 5, 4, 1}
		evaRes0, evaRes1 := params.ModelParallel()
		label1 := LoadLabel("label/label_1.csv")
		RMSE1 := CompareRMSE(evaRes0, label1)
		fmt.Println("Task 1 RMSE: ", RMSE1)

		label5 := LoadLabel("label/label_5.csv")
		AUC5 := CompareAUC(evaRes1, label5)
		fmt.Println("Task 5 AUC: ", AUC5)
	})
	t.Run("Model XBGLasso Parallel for task 1 and task 2", func(t *testing.T) {
		params := ModelParallelParams{2, 1, 2, 4, 1}
		evaRes0, evaRes1 := params.ModelParallel()
		label1 := LoadLabel("label/label_1.csv")
		RMSE1 := CompareRMSE(evaRes0, label1)
		fmt.Println("Task 1 RMSE: ", RMSE1)

		label2 := LoadLabel("label/label_2.csv")
		RMSE2 := CompareRMSE(evaRes1, label2)
		fmt.Println("Task 2 RMSE: ", RMSE2)
	})
	t.Run("Model XBGLasso Parallel for task 1 and task 4", func(t *testing.T) {
		params := ModelParallelParams{2, 1, 4, 4, 1}
		evaRes0, evaRes1 := params.ModelParallel()
		label1 := LoadLabel("label/label_1.csv")
		RMSE1 := CompareRMSE(evaRes0, label1)
		fmt.Println("Task 1 RMSE: ", RMSE1)

		label4 := LoadLabel("label/label_4.csv")
		RMSE4 := CompareRMSE(evaRes1, label4)
		fmt.Println("Task 4 RMSE: ", RMSE4)
	})
	t.Run("TASK 4 Plaintext evaluate", func(t *testing.T) {
		plaintextOmega := LoadWeightPlain("weight/weight4.csv")
		plaintextTheta := LoadWeightPlain("weight/weight4.csv")
		label := LoadLabel("label/label_4.csv")
		evaReSlice := make([]float64, 0)
		for i := 1; i < 199; i++ {
			evaRes := PlainEvaluate2(plaintextOmega, plaintextTheta, LoadDataPlain(i), 4)
			evaReSlice = append(evaReSlice, evaRes)
		}

		RMSE := CompareRMSE(evaReSlice, label)
		fmt.Println("RMSE: ", RMSE)
	})
}
