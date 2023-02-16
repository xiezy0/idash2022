package main

import (
	"fmt"
	"strconv"
	"time"
)

type ModelParams struct {
	model        int
	task         int
	threadNum    int
	encodeParams float64
}

type ModelParallelParams struct {
	model        int
	task0, task1 int
	threadNum    int
	encodeParams float64
}

// Model evaluation
func (modelParams *ModelParams) Model() (evaRes []float64) {
	time1 := time.Now()
	// load intercept
	interSlice := LoadIntercept(modelParams.encodeParams)

	// load weight for different model
	switch modelParams.model {
	case 1:
		lassoPlainWeights = LoadWeight("weight/weight"+strconv.Itoa(modelParams.task)+".csv", modelParams.encodeParams)
		lassoPlainIntercept = interSlice[modelParams.task-1]
	case 2:
		XGBLassoPlainOmega = LoadWeight("weight/weight"+strconv.Itoa(modelParams.task)+".csv", modelParams.encodeParams)
		XGBLassoPlainTheta = LoadWeight("weight/weight"+strconv.Itoa(modelParams.task)+".csv", modelParams.encodeParams)
		XGBLassoPlainIntercept = interSlice[modelParams.task-1]
	default:
		panic("model load error")
	}

	// load the rest data
	plainDataSlice = LoadDataRenew()
	timeLoad := time.Since(time1)
	fmt.Println("load", timeLoad)

	// evaluate model in ciphertext by multi goroutine
	switch modelParams.model {
	case 1:
		lasso := LassoParams{modelParams, lassoPlainWeights, plainDataSlice, lassoPlainIntercept}
		evaRes = lasso.LassoThread()
	case 2:
		lasso := XGBLassoParams{modelParams, XGBLassoPlainOmega, XGBLassoPlainTheta, plainDataSlice, XGBLassoPlainIntercept}
		evaRes = lasso.XGBLassoThread()
	}
	return
}

// ModelParallel model evaluation
func (modelParams *ModelParallelParams) ModelParallel() (evaRes0, evaRes1 []float64) {
	time1 := time.Now()
	// load intercept
	interSlice := LoadIntercept(modelParams.encodeParams)

	// load weight for different model
	switch modelParams.model {
	case 1:
		lassoPlainWeights0 = LoadWeight("weight/weight"+strconv.Itoa(modelParams.task0)+".csv", modelParams.encodeParams)
		lassoPlainWeights1 = LoadWeight("weight/weight"+strconv.Itoa(modelParams.task1)+".csv", modelParams.encodeParams)
		lassoPlainIntercept0 = interSlice[modelParams.task0-1]
		lassoPlainIntercept1 = interSlice[modelParams.task1-1]
	case 2:
		XGBLassoPlainOmega0 = LoadWeight("weight/weight"+strconv.Itoa(modelParams.task0)+".csv", modelParams.encodeParams)
		XGBLassoPlainOmega1 = LoadWeight("weight/weight"+strconv.Itoa(modelParams.task1)+".csv", modelParams.encodeParams)
		XGBLassoPlainTheta0 = LoadWeight("weight/weight"+strconv.Itoa(modelParams.task0)+".csv", modelParams.encodeParams)
		XGBLassoPlainTheta1 = LoadWeight("weight/weight"+strconv.Itoa(modelParams.task1)+".csv", modelParams.encodeParams)
		XGBLassoPlainIntercept0 = interSlice[modelParams.task0-1]
		XGBLassoPlainIntercept1 = interSlice[modelParams.task1-1]
	default:
		panic("model load error")
	}

	// load the rest data
	plainDataSlice = LoadDataRenew()
	timeLoad := time.Since(time1)
	fmt.Println("load", timeLoad)

	// evaluate model in ciphertext by multi goroutine
	switch modelParams.model {
	case 1:
		lasso := LassoParallelParams{modelParams, lassoPlainWeights0, lassoPlainWeights1, plainDataSlice, lassoPlainIntercept0, lassoPlainIntercept1}
		evaRes0, evaRes1 = lasso.LassoThreadParallel()
	case 2:
		xlasso := XGBLassoParallelParams{modelParams, XGBLassoPlainOmega0, XGBLassoPlainOmega1, XGBLassoPlainTheta0, XGBLassoPlainTheta1, plainDataSlice, XGBLassoPlainIntercept0, XGBLassoPlainIntercept1}
		evaRes0, evaRes1 = xlasso.XGBLassoThreadParallel()
	}

	return
}

// PlainEvaluate evaluate model (lasso) in plaintext, used to compare with the model in ciphertext
func PlainEvaluate(plaintextWeights []float64, plaintextData []float64, number int) float64 {
	result := float64(0)
	for i, value := range plaintextWeights {
		weightData := value * plaintextData[i]
		result = result + weightData
	}
	switch number {
	case 1:
		result = result - 0.00856
	case 2:
		result = result - 0.0199
	case 3:
		result = result + 0.0693
	default:
	}
	return result
}

func PlainEvaluate2(plaintextOmega, plaintextTheta, plaintextData []float64, number int) float64 {
	res := float64(0)
	for i, value := range plaintextTheta {
		omegaX2 := plaintextData[i] * plaintextData[i] * plaintextOmega[i]
		thetaX := plaintextData[i] * value
		res = res + omegaX2 + thetaX
	}
	switch number {
	case 1:
		res = res - 0.00856
	case 2:
		res = res - 0.0199
	case 3:
		res = res + 0.0693
	default:
	}
	return res
}
