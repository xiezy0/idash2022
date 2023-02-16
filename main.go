package main

import (
	"fmt"
	"strconv"
	"sync"
	"time"
)

var (
	lassoWeights   [][][]float64
	lassoIntercept [][]float64
)

//func init() {
//	time0 := time.Now()
//	encodeParmas := float64(1)
//	lassoWeights, lassoIntercept = make([][][]float64, 5), make([][]float64, 5)
//
//	lassoWeights[0] = LoadWeight("weight/weight"+strconv.Itoa(1)+".csv", encodeParmas)
//	lassoWeights[1] = LoadWeight("weight/weight"+strconv.Itoa(2)+".csv", encodeParmas)
//	lassoWeights[2] = LoadWeight("weight/weight"+strconv.Itoa(3)+".csv", encodeParmas)
//	lassoWeights[3] = LoadWeight("weight/weight"+strconv.Itoa(4)+".csv", encodeParmas)
//	lassoWeights[4] = LoadWeight("weight/weight"+strconv.Itoa(5)+".csv", encodeParmas)
//
//	//fmt.Println("timeWeight", time.Since(time0))
//	//
//	//time1 := time.Now()
//	interSlice := LoadIntercept(1)
//
//	lassoIntercept[0] = interSlice[0]
//	lassoIntercept[1] = interSlice[1]
//	lassoIntercept[2] = interSlice[2]
//	lassoIntercept[3] = interSlice[3]
//	lassoIntercept[4] = interSlice[4]
//
//	//time2 := time.Now()
//	plainDataSlice = LoadDataRenew()
//	fmt.Println("timeLoad single goroutine", time.Since(time0))
//}

func init() {
	time0 := time.Now()
	encodeParmas := float64(1)
	lassoWeights, lassoIntercept = make([][][]float64, 5), make([][]float64, 5)

	var wg sync.WaitGroup
	wg.Add(7)
	for i := 0; i < 7; i++ {
		go func(threadId int) {
			switch threadId {
			case 5:
				plainDataSlice = LoadDataRenew()
			case 6:
				lassoIntercept = LoadIntercept(1)
			default:
				lassoWeights[threadId] = LoadWeight("weight/weight"+strconv.Itoa(threadId+1)+".csv", encodeParmas)
			}
			wg.Done()
		}(i)

	}
	wg.Wait()
	fmt.Println(len(lassoWeights[0]))
	fmt.Println("timeLoad seven goroutine", time.Since(time0))
}

func main() {
	time0 := time.Now()
	params1 := &ModelParallelParams{1, 1, 2, 1020, 1}
	lasso1 := LassoParallelParams{params1, lassoWeights[0], lassoWeights[1],
		plainDataSlice, lassoIntercept[0], lassoIntercept[1]}
	evaRes1, evaRes2 := lasso1.LassoThreadParallel()

	params2 := &ModelParams{1, 3, 1020, 1}
	lasso := LassoParams{params2, lassoWeights[2], plainDataSlice, lassoIntercept[2]}
	evaRes3 := lasso.LassoThread()

	params3 := &ModelParallelParams{1, 4, 5, 1020, 1}
	lasso2 := LassoParallelParams{params3, lassoWeights[3], lassoWeights[4],
		plainDataSlice, lassoIntercept[3], lassoIntercept[4]}
	evaRes4, evaRes5 := lasso2.LassoThreadParallel()
	timeAll := time.Since(time0)
	fmt.Println(timeAll)

	TraceMemStats()
	label1 := LoadLabel("label/label_1.csv")
	RMSE1 := CompareRMSE(evaRes1, label1)
	fmt.Println("Task 1 RMSE: ", RMSE1)

	label2 := LoadLabel("label/label_2.csv")
	RMSE2 := CompareRMSE(evaRes2, label2)
	fmt.Println("Task 2 RMSE: ", RMSE2)

	label3 := LoadLabel("label/label_3.csv")
	RMSE3 := CompareRMSE(evaRes3, label3)
	fmt.Println("Task 3 RMSE: ", RMSE3)

	label4 := LoadLabel("label/label_4.csv")
	AUC4 := CompareAUC(evaRes4, label4)
	fmt.Println("Task 4 AUC: ", AUC4)

	label5 := LoadLabel("label/label_5.csv")
	AUC5 := CompareAUC(evaRes5, label5)
	fmt.Println("Task 5 AUC: ", AUC5)
}
