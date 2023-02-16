package main

import (
	"fmt"
	"math"
	"runtime"
	"strconv"
	"time"

	"github.com/tuneinsight/lattigo/v3/ckks"
)

var (
	// ----- Lasso & XGB Params ----- //
	lassoPlainWeights   [][]float64
	lassoPlainIntercept []float64

	XGBLassoPlainOmega     [][]float64
	XGBLassoPlainTheta     [][]float64
	XGBLassoPlainIntercept []float64

	plainDataSlice [][]float64

	lassoPlainWeights0 [][]float64
	lassoPlainWeights1 [][]float64

	lassoPlainIntercept0 []float64
	lassoPlainIntercept1 []float64

	XGBLassoPlainOmega0 [][]float64
	XGBLassoPlainOmega1 [][]float64
	XGBLassoPlainTheta0 [][]float64
	XGBLassoPlainTheta1 [][]float64

	XGBLassoPlainIntercept0 []float64
	XGBLassoPlainIntercept1 []float64
)

var (
	timeEncodeEncrypt time.Duration
	timeDotMul        time.Duration
	timeDecrypt       time.Duration
	timeEvaluate      time.Duration
	timeDotAdd        time.Duration
	timeRelin         time.Duration
	timeRescale       time.Duration
	timeDotSum        time.Duration
)

// RunTimed test the time lost in goroutine
func RunTimed(f func()) time.Duration {
	start := time.Now()
	f()
	return time.Since(start)
}

// TraceMemStats test the memory lost and garbage collection
func TraceMemStats() {
	var ms runtime.MemStats
	runtime.ReadMemStats(&ms)
	fmt.Printf("Alloc:%d(MB) HeapIdle:%d(MB) HeapReleased:%d(MB)\n", ms.Alloc/1048576, ms.HeapIdle/1048576, ms.HeapReleased/1048576)
}

// String2Float64 string ------> float64
func String2Float64(strArr []string) []float64 {
	res := make([]float64, len(strArr))
	for index, val := range strArr {
		res[index], _ = strconv.ParseFloat(val, 64)
	}
	return res
}

// CompareRMSE evaluate root-mean-square error in continuous model
func CompareRMSE(evaRes []float64, label []float64) (RMSError float64) {
	var value float64
	value = 0
	for i, labelValue := range label {
		valueOne := math.Pow(evaRes[i]-labelValue, 2)
		value = value + valueOne
	}
	RMSError = value / float64(len(label))
	RMSError = math.Sqrt(RMSError)
	return
}

// CompareAUC evaluate accuracy in discrete model
func CompareAUC(evaRes []float64, label []float64) float64 {
	num := 0
	for i, labelValue := range label {
		if evaRes[i] == labelValue {
			num++
		}
	}
	return float64(num) / 198
}

// Sigmoid Linear approximation function TODO: 后续优化
func (params *Ckksparams) Sigmoid(ciphertextIn *ckks.Ciphertext, encodeParams float64) *ckks.Ciphertext {
	weight := make([]float64, 0)
	intercept := make([]float64, 0)
	for i := 0; i < 198; i++ {
		weight = append(weight, 0.03333*encodeParams)
		intercept = append(intercept, 0.5*encodeParams)
	}
	sigmoidWeight := params.Encrypt(weight)
	sigmoidIntercept := params.Encrypt(intercept)
	cipherTestOut := params.Mul(sigmoidWeight, ciphertextIn)
	cipherTestOut = params.Add(cipherTestOut, sigmoidIntercept)

	return cipherTestOut
}

// SigmoidParallel Linear approximation function TODO: 后续优化
func (params *Ckksparams) SigmoidParallel(ciphertextIn *ckks.Ciphertext, encodeParams float64) *ckks.Ciphertext {
	weight := make([]float64, 0)
	intercept := make([]complex128, 0)
	for i := 0; i < 198; i++ {
		weight = append(weight, 0.03333*encodeParams)
		intercept = append(intercept, complex(0.5*encodeParams, 0.5*encodeParams))
	}
	sigmoidWeight := params.Encrypt(weight)
	sigmoidIntercept := params.Encrypt(intercept)
	cipherTestOut := params.Mul(sigmoidWeight, ciphertextIn)
	cipherTestOut = params.Add(cipherTestOut, sigmoidIntercept)

	return cipherTestOut
}

// lassoThreadMul evaluate ω * x
func (params *Ckksparams) lassoThreadMul(weightCipher, dataCipher *ckks.Ciphertext) *ckks.Ciphertext {
	evaluator := params.evaluator.ShallowCopy()
	ciphertext0 := evaluator.MulNew(weightCipher, dataCipher)
	return ciphertext0
}

// XGBLassoThreadMul evaluate θ * x & ω * x^2
func (params *Ckksparams) XGBLassoThreadMul(omegaCipher, thetaCipher, dataCipher *ckks.Ciphertext) (ciphertext0, ciphertext1 *ckks.Ciphertext) {
	evaluator := params.evaluator.ShallowCopy()
	// θ * x
	ciphertext0 = evaluator.MulNew(thetaCipher, dataCipher)
	// x^2 (faster)
	ciphertext1 = evaluator.MulNew(dataCipher, dataCipher)
	// ω * x^2
	ciphertext1 = evaluator.Mul2DegreeNew(omegaCipher, ciphertext1)
	return
}

// LassoDotProduct add + Re-linearize + Re-scalar and Inner-sum
func (params *Ckksparams) LassoDotProduct(ciphertext *ckks.Ciphertext, cipherSlice []*ckks.Ciphertext) *ckks.Ciphertext {
	time0 := time.Now()
	for i := 0; i < 1020; i++ {
		ciphertext = params.Add(ciphertext, cipherSlice[i])
	}
	timeDotAdd = time.Since(time0)
	//fmt.Println(params.Decrypt(ciphertext))
	time1 := time.Now()
	params.evaluator.Relinearize(ciphertext, ciphertext)
	timeRelin = time.Since(time1)

	time2 := time.Now()
	err := params.evaluator.Rescale(ciphertext, params.params.DefaultScale(), ciphertext)
	timeRescale = time.Since(time2)
	if err != nil {
		panic("rescale error In DotProduct")
	}

	time3 := time.Now()
	params.evaluator.InnerSumLog(ciphertext, 198, 20, ciphertext)
	timeDotSum = time.Since(time3)

	return ciphertext
}

// XGBLassoDotProduct add + Re-linearize + Re-scalar and Inner-sum
func (params *Ckksparams) XGBLassoDotProduct(cipherThetaXSlice, cipherOmegaX2Slice []*ckks.Ciphertext) *ckks.Ciphertext {
	cipherThetaX := params.Encrypt([]float64{0})
	// cipherOmegaX2 := params.Encrypt([]float64{0})
	cipherOmegaX2 := params.evaluator.MulNew(params.Encrypt([]float64{0}), params.Encrypt([]float64{0}))
	cipherOmegaX2 = params.evaluator.Mul2DegreeNew(params.Encrypt([]float64{0}), cipherOmegaX2)
	time0 := time.Now()
	for i := 0; i < 1020; i++ {
		cipherThetaX = params.Add(cipherThetaX, cipherThetaXSlice[i])
		cipherOmegaX2 = params.Add(cipherOmegaX2, cipherOmegaX2Slice[i])
	}
	timeDotAdd = time.Since(time0)
	// fmt.Println(params.Decrypt(cipherOmegaX2))

	time1 := time.Now()
	params.evaluator.Relinearize(cipherThetaX, cipherThetaX)
	params.evaluator.RelinearizeDegree(cipherOmegaX2, cipherOmegaX2)
	timeRelin = time.Since(time1)

	time3 := time.Now()
	err := params.evaluator.Rescale(cipherThetaX, params.params.DefaultScale(), cipherThetaX)
	err = params.evaluator.Rescale(cipherOmegaX2, params.params.DefaultScale(), cipherOmegaX2)
	timeRescale = time.Since(time3)

	cipherResult := params.Add(cipherThetaX, cipherOmegaX2)
	// fmt.Println(params.Decrypt(cipherResult))
	if err != nil {
		panic("rescale error In DotProduct")
	}

	time2 := time.Now()
	params.evaluator.InnerSumLog(cipherResult, 198, 20, cipherResult)
	timeDotSum = time.Since(time2)

	return cipherResult
}

// ResFormat decode the result with encodeParams if task > 3 /(encodeParams * 10^4), if task <= 3 / encodeParams
func ResFormat(Result []complex128, task int, encodeParams float64, ri bool) (evaRes []float64) {
	evaRes = make([]float64, 0)

	if task > 3 {
		for i := 0; i < 198; i++ {
			if ri {
				if real(Result[i]) >= 0.5 {
					evaRes = append(evaRes, 1)
				} else {
					evaRes = append(evaRes, 0)
				}
			} else {
				if imag(Result[i]) >= 0.5 {
					evaRes = append(evaRes, 1)
				} else {
					evaRes = append(evaRes, 0)
				}
			}
		}
	} else {
		for i := 0; i < 198; i++ {
			if ri {
				res0 := real(Result[i]) / encodeParams
				evaRes = append(evaRes, res0)
			} else {
				res1 := imag(Result[i]) / encodeParams
				evaRes = append(evaRes, res1)
			}
		}
	}
	return
}
