package main

import (
	"fmt"
	"github.com/tuneinsight/lattigo/v3/ckks"
	"sync"
	"time"
)

type LassoParams struct {
	modelParams         *ModelParams
	plaintextWeights    [][]float64
	plaintextDataSlice  [][]float64
	plaintextInterSlice []float64
}

type LassoParallelParams struct {
	modelParams                                *ModelParallelParams
	plaintextWeights0, plaintextWeights1       [][]float64
	plaintextDataSlice                         [][]float64
	plaintextInterSlice0, plaintextInterSlice1 []float64
}

type Lasso interface {
	Lasso() (evaRes []float64)
	LassoThread() (evaRes []float64)
}

type LassoParallel interface {
	LassoThreadParallel() (evaRes0, evaRes1 []float64)
}

// Lasso evaluate model in ciphertext by single goroutine
func (lasso *LassoParams) Lasso() (evaRes []float64) {
	resChannel := make(chan []complex128, 1)
	runTimed := RunTimed(func() {
		evaRes = make([]float64, 0)
		var ciphertext0 *ckks.Ciphertext
		params := GenKey(ckks.PN13QP218SEAL)
		ciphertextInter := params.Encrypt(lasso.plaintextInterSlice)
		ciphertext := params.Encrypt([]float64{0})
		for i := 0; i < 1020; i++ {
			weight := params.Encrypt(lasso.plaintextWeights[i])
			data := params.Encrypt(lasso.plaintextDataSlice[i])
			if i == 1019 {
				//fmt.Println(len(plaintextWeights[i]))
				//fmt.Println(len(plaintextDataSlice[i]))
				ciphertext0 = params.InnerSumBlock(weight, data, 198, 10)
			} else {
				ciphertext0 = params.InnerSumBlock(weight, data, 198, 20)
			}
			ciphertext = params.Add(ciphertext, ciphertext0)
		}
		ciphertext = params.Add(ciphertext, ciphertextInter)
		// sigmoid 函数
		if lasso.modelParams.task > 3 {
			ciphertext = params.Sigmoid(ciphertext, lasso.modelParams.encodeParams)
		}

		result := params.Decrypt(ciphertext)
		TraceMemStats()
		resChannel <- result
	})

	fmt.Println("runtime: ", runTimed)
	Result := <-resChannel
	if lasso.modelParams.task > 3 {
		for i := 0; i < 198; i++ {
			res := real(Result[i])
			if res >= 0.5 {
				evaRes = append(evaRes, 1)
			} else {
				evaRes = append(evaRes, 0)
			}
		}
	} else {
		for i := 0; i < 198; i++ {
			res := real(Result[i]) / lasso.modelParams.encodeParams
			evaRes = append(evaRes, res)
		}
	}

	return
}

// LassoThread evaluate model (lasso) in ciphertext by multi goroutine
func (lasso *LassoParams) LassoThread() (evaRes []float64) {
	var wg sync.WaitGroup
	var lock sync.Mutex
	wg.Add(lasso.modelParams.threadNum)

	threadBatch := 1020 / lasso.modelParams.threadNum
	timeAll := time.Now()
	weightCipherSlice, dataCipherSlice, cipherSlice := make([]*ckks.Ciphertext, 0), make([]*ckks.Ciphertext, 0), make([]*ckks.Ciphertext, 0)
	params := GenKey(ckks.PN13QP218SEAL)
	ciphertextInter := params.Encrypt(lasso.plaintextInterSlice)
	ciphertext := params.Encrypt([]float64{0})
	for i := 0; i < 1020; i++ {
		weightCipherSlice = append(weightCipherSlice, new(ckks.Ciphertext))
		dataCipherSlice = append(dataCipherSlice, new(ckks.Ciphertext))
	}
	// ------- encode & encrypt ------- //
	time0 := time.Now()
	for i := 0; i < lasso.modelParams.threadNum; i++ {
		go func(threadId int) {
			for batchNum := 0; batchNum < threadBatch; batchNum++ {
				encoder := params.encoder.ShallowCopy()
				encryptor := params.encryptor.ShallowCopy()
				encryptorSk := params.encryptorSk.ShallowCopy()
				weightPlain := encoder.EncodeSlotsNew(lasso.plaintextWeights[threadId*threadBatch+batchNum], params.params.MaxLevel(), params.params.DefaultScale(), params.params.LogSlots())
				dataPlain := encoder.EncodeSlotsNew(lasso.plaintextDataSlice[threadId*threadBatch+batchNum], params.params.MaxLevel(), params.params.DefaultScale(), params.params.LogSlots())
				weightCipherSlice[threadId*threadBatch+batchNum] = encryptor.EncryptNew(weightPlain)

				dataCipherSlice[threadId*threadBatch+batchNum] = encryptorSk.EncryptNew(dataPlain)
				//if threadId == 0 {
				//	testWeight := encryptor.EncryptNew(weightPlain)
				//	testData := encryptor.EncryptNew(dataPlain)
				//	fmt.Println(len(testWeight.Value[0].Coeffs[0]) + len(testWeight.Value[0].Coeffs[1]) + len(testWeight.Value[0].Coeffs[2]) + len(testWeight.Value[0].Coeffs[3]))
				//	fmt.Println(unsafe.Sizeof(testWeight.Value[0].Coeffs[0]))
				//	fmt.Println(unsafe.Sizeof(testData.Value[0].Coeffs))
				//}
			}
			wg.Done()
		}(i)
	}
	wg.Wait()
	wg.Add(lasso.modelParams.threadNum)
	timeEncodeEncrypt = time.Since(time0)

	time1 := time.Now()
	for i := 0; i < lasso.modelParams.threadNum; i++ {
		go func(threadId int) {
			for batchNum := 0; batchNum < threadBatch; batchNum++ {
				ciphertext0 := params.lassoThreadMul(weightCipherSlice[threadId*threadBatch+batchNum], dataCipherSlice[threadId*threadBatch+batchNum])
				lock.Lock()
				cipherSlice = append(cipherSlice, ciphertext0)
				lock.Unlock()
			}
			wg.Done()
		}(i)
	}

	wg.Wait()
	timeDotMul = time.Since(time1)

	cipherRes := params.LassoDotProduct(ciphertext, cipherSlice)
	ciphertext = params.Add(cipherRes, ciphertextInter)

	// sigmoid 函数
	if lasso.modelParams.task > 3 {
		ciphertext = params.Sigmoid(ciphertext, lasso.modelParams.encodeParams)
	}

	time2 := time.Now()
	result := params.Decrypt(ciphertext)
	timeDecrypt = time.Since(time2)

	timeEvaluate = time.Since(timeAll)
	fmt.Printf("encode & encrypt time:%v || DotMul time:%v || relinearize time:%v || rescale time:%v || decrypt time: %v \n",
		timeEncodeEncrypt, timeDotMul+timeDotSum+timeDotAdd, timeRelin, timeRescale, timeDecrypt)
	fmt.Println("runtime:", timeEvaluate)
	TraceMemStats()

	evaRes = ResFormat(result, lasso.modelParams.task, lasso.modelParams.encodeParams, true)
	return
}

// LassoThreadParallel evaluate model (lasso) in ciphertext by multi goroutine
func (lasso *LassoParallelParams) LassoThreadParallel() (evaRes0, evaRes1 []float64) {
	var wg sync.WaitGroup
	var lock sync.Mutex
	wg.Add(lasso.modelParams.threadNum)

	threadBatch := 1020 / lasso.modelParams.threadNum
	// ------- init ------- //
	timeAll := time.Now()
	params := GenKey(ckks.PN13QP218SEAL)
	ciphertext := params.Encrypt([]float64{0})
	plaintextWeights := make([][]complex128, 0)
	plaintextInterSlice := make([]complex128, 0)
	weightCipherSlice, dataCipherSlice, cipherSlice := make([]*ckks.Ciphertext, 0), make([]*ckks.Ciphertext, 0), make([]*ckks.Ciphertext, 0)
	result0, result1 := make([]complex128, 0), make([]complex128, 0)

	for i := 0; i < 1020; i++ {
		weightCipherSlice = append(weightCipherSlice, new(ckks.Ciphertext))
		dataCipherSlice = append(dataCipherSlice, new(ckks.Ciphertext))
		plaintextWeights = append(plaintextWeights, []complex128{})
	}
	// ------- encode & encrypt ------- //
	// cut Slice and parallel two task weight and intercept to single complex128
	for i := 0; i < 1020; i++ {
		for num := 0; num < len(lasso.plaintextWeights0[i]); num++ {
			weight := complex(lasso.plaintextWeights0[i][num], lasso.plaintextWeights1[i][num])
			plaintextWeights[i] = append(plaintextWeights[i], weight)
		}
	}
	for i := 0; i < 198; i++ {
		intercept := complex(lasso.plaintextInterSlice0[i], lasso.plaintextInterSlice1[i])
		plaintextInterSlice = append(plaintextInterSlice, intercept)
	}

	ciphertextInter := params.Encrypt(plaintextInterSlice)

	// encrypt weight and data
	time0 := time.Now()
	for i := 0; i < lasso.modelParams.threadNum; i++ {
		go func(threadId int) {
			for batchNum := 0; batchNum < threadBatch; batchNum++ {
				encoder := params.encoder.ShallowCopy()
				encryptor := params.encryptor.ShallowCopy()
				encryptorSk := params.encryptorSk.ShallowCopy()
				weightPlain := encoder.EncodeSlotsNew(plaintextWeights[threadId*threadBatch+batchNum], params.params.MaxLevel(), params.params.DefaultScale(), params.params.LogSlots())
				dataPlain := encoder.EncodeSlotsNew(lasso.plaintextDataSlice[threadId*threadBatch+batchNum], params.params.MaxLevel(), params.params.DefaultScale(), params.params.LogSlots())
				weightCipherSlice[threadId*threadBatch+batchNum] = encryptor.EncryptNew(weightPlain)
				dataCipherSlice[threadId*threadBatch+batchNum] = encryptorSk.EncryptNew(dataPlain)
			}
			wg.Done()
		}(i)
	}
	wg.Wait()
	wg.Add(lasso.modelParams.threadNum)
	timeEncodeEncrypt = time.Since(time0)

	time1 := time.Now()
	// ------- Dot product --------//
	// Multiply the weight and data in 1020 goroutine
	for i := 0; i < lasso.modelParams.threadNum; i++ {
		go func(threadId int) {
			for batchNum := 0; batchNum < threadBatch; batchNum++ {
				ciphertext0 := params.lassoThreadMul(weightCipherSlice[threadId*threadBatch+batchNum], dataCipherSlice[threadId*threadBatch+batchNum])
				lock.Lock()
				cipherSlice = append(cipherSlice, ciphertext0)
				lock.Unlock()
			}
			wg.Done()
		}(i)
	}

	wg.Wait()
	timeDotMul = time.Since(time1)
	// add relinearize rescale and rotation
	cipherRes := params.LassoDotProduct(ciphertext, cipherSlice)
	ciphertext = params.Add(cipherRes, ciphertextInter)

	// ----------- run sigmoid function and decrypt ---------//
	time2 := time.Now()
	if lasso.modelParams.task0 > 3 && lasso.modelParams.task1 > 3 {
		ciphertext = params.SigmoidParallel(ciphertext, lasso.modelParams.encodeParams)
		result0 = params.Decrypt(ciphertext)
		result1 = result0
	} else if lasso.modelParams.task0 < 3 && lasso.modelParams.task1 > 3 {
		result0 = params.Decrypt(ciphertext)
		ciphertext = params.SigmoidParallel(ciphertext, lasso.modelParams.encodeParams)
		result1 = params.Decrypt(ciphertext)
	} else if lasso.modelParams.task0 < 3 && lasso.modelParams.task1 < 3 {
		result0 = params.Decrypt(ciphertext)
		result1 = result0
	}
	timeDecrypt = time.Since(time2)

	timeEvaluate = time.Since(timeAll)
	fmt.Printf("encode & encrypt time:%v || DotMul time:%v || relinearize time:%v || rescale time:%v || decrypt + sigmoid time: %v \n",
		timeEncodeEncrypt, timeDotMul+timeDotSum+timeDotAdd, timeRelin, timeRescale, timeDecrypt)
	fmt.Println("runTime: ", timeEvaluate)
	TraceMemStats()
	evaRes0 = ResFormat(result0, lasso.modelParams.task0, lasso.modelParams.encodeParams, true)
	evaRes1 = ResFormat(result1, lasso.modelParams.task1, lasso.modelParams.encodeParams, false)
	return
}
