package main

import (
	"fmt"
	"github.com/tuneinsight/lattigo/v3/ckks"
	"sync"
	"time"
)

type XGBLassoParams struct {
	modelParams     *ModelParams
	plainOmega      [][]float64
	plainTheta      [][]float64
	plainDataSlice  [][]float64
	plainInterSlice []float64
}

type XGBLassoParallelParams struct {
	modelParams                        *ModelParallelParams
	plainOmega0, plainOmega1           [][]float64
	plainTheta0, plainTheta1           [][]float64
	plainDataSlice                     [][]float64
	plainInterSlice0, plainInterSlice1 []float64
}

type XGBLasso interface {
	XGBLassoThread() (evaRes []float64)
}

type XGBLassoParallel interface {
	XGBLassoThreadParallel() (evaRes0, evaRes1 []float64)
}

// XGBLassoThread evaluate model (XGB & lasso) in ciphertext by multi goroutine
func (xlasso *XGBLassoParams) XGBLassoThread() (evaRes []float64) {
	var wg sync.WaitGroup
	var lock sync.Mutex
	wg.Add(xlasso.modelParams.threadNum)
	threadBatch := 1020 / xlasso.modelParams.threadNum

	timeAll := time.Now()
	omegaCipherSlice, thetaCipherSlice, dataCipherSlice, cipherThetaXSlice, cipherOmegaX2Slice := make([]*ckks.Ciphertext, 0), make([]*ckks.Ciphertext, 0), make([]*ckks.Ciphertext, 0), make([]*ckks.Ciphertext, 0), make([]*ckks.Ciphertext, 0)
	params := GenKey(ckks.PN13QP218SEAL)
	ciphertextInter := params.Encrypt(xlasso.plainInterSlice)

	for i := 0; i < 1020; i++ {
		omegaCipherSlice = append(omegaCipherSlice, new(ckks.Ciphertext))
		thetaCipherSlice = append(thetaCipherSlice, new(ckks.Ciphertext))
		dataCipherSlice = append(dataCipherSlice, new(ckks.Ciphertext))
	}

	// Encode & Encrypt
	time0 := time.Now()
	for i := 0; i < xlasso.modelParams.threadNum; i++ {
		go func(threadId int) {
			for batchNum := 0; batchNum < threadBatch; batchNum++ {
				encoder := params.encoder.ShallowCopy()
				encryptor := params.encryptor.ShallowCopy()
				encryptorSk := params.encryptorSk.ShallowCopy()
				omegaPlain := encoder.EncodeSlotsNew(xlasso.plainOmega[threadId*threadBatch+batchNum], params.params.MaxLevel(), params.params.DefaultScale(), params.params.LogSlots())
				thetaPlain := encoder.EncodeSlotsNew(xlasso.plainTheta[threadId*threadBatch+batchNum], params.params.MaxLevel(), params.params.DefaultScale(), params.params.LogSlots())
				dataPlain := encoder.EncodeSlotsNew(plainDataSlice[threadId*threadBatch+batchNum], params.params.MaxLevel(), params.params.DefaultScale(), params.params.LogSlots())
				omegaCipherSlice[threadId*threadBatch+batchNum] = encryptor.EncryptNew(omegaPlain)
				thetaCipherSlice[threadId*threadBatch+batchNum] = encryptor.EncryptNew(thetaPlain)
				dataCipherSlice[threadId*threadBatch+batchNum] = encryptorSk.EncryptNew(dataPlain)
			}

			wg.Done()
		}(i)
	}
	wg.Wait()
	wg.Add(xlasso.modelParams.threadNum)
	timeEncodeEncrypt = time.Since(time0)

	time1 := time.Now()
	for i := 0; i < xlasso.modelParams.threadNum; i++ {
		go func(threadId int) {
			for batchNum := 0; batchNum < threadBatch; batchNum++ {
				cipherThetaX, cipherOmegaX2 := params.XGBLassoThreadMul(omegaCipherSlice[threadId*threadBatch+batchNum], thetaCipherSlice[threadId*threadBatch+batchNum], dataCipherSlice[threadId*threadBatch+batchNum])
				lock.Lock()
				cipherThetaXSlice = append(cipherThetaXSlice, cipherThetaX)
				cipherOmegaX2Slice = append(cipherOmegaX2Slice, cipherOmegaX2)
				lock.Unlock()
			}

			wg.Done()
		}(i)
	}
	wg.Wait()
	timeDotMul = time.Since(time1)

	cipherRes := params.XGBLassoDotProduct(cipherThetaXSlice, cipherOmegaX2Slice)
	ciphertext := params.Add(cipherRes, ciphertextInter)

	// sigmoid 函数
	if xlasso.modelParams.task > 3 {
		ciphertext = params.Sigmoid(ciphertext, xlasso.modelParams.encodeParams)
	}

	time2 := time.Now()
	result := params.Decrypt(ciphertext)
	timeDecrypt = time.Since(time2)
	timeEvaluate = time.Since(timeAll)

	fmt.Printf("encode & encrypt time:%v || DotMul time:%v || relinearize time:%v || rescale time:%v || decrypt time: %v \n",
		timeEncodeEncrypt, timeDotMul+timeDotSum+timeDotAdd, timeRelin, timeRescale, timeDecrypt)
	fmt.Println("runtime:", timeEvaluate)
	TraceMemStats()

	Result := result
	evaRes = ResFormat(Result, xlasso.modelParams.task, xlasso.modelParams.encodeParams, true)
	return
}

// XGBLassoThreadParallel evaluate model (XGB & lasso) in ciphertext by multi goroutine
func (xlasso *XGBLassoParallelParams) XGBLassoThreadParallel() (evaRes0, evaRes1 []float64) {
	var wg sync.WaitGroup
	var lock sync.Mutex
	wg.Add(xlasso.modelParams.threadNum)
	threadBatch := 1020 / xlasso.modelParams.threadNum

	timeAll := time.Now()

	// ------- init ------- //
	omegaCipherSlice, thetaCipherSlice, dataCipherSlice, cipherThetaXSlice, cipherOmegaX2Slice := make([]*ckks.Ciphertext, 0), make([]*ckks.Ciphertext, 0), make([]*ckks.Ciphertext, 0), make([]*ckks.Ciphertext, 0), make([]*ckks.Ciphertext, 0)
	plaintextOmega, plaintextTheta := make([][]complex128, 0), make([][]complex128, 0)
	plaintextInterSlice := make([]complex128, 0)
	params := GenKey(ckks.PN13QP218SEAL)

	for i := 0; i < 1020; i++ {
		omegaCipherSlice = append(omegaCipherSlice, new(ckks.Ciphertext))
		thetaCipherSlice = append(thetaCipherSlice, new(ckks.Ciphertext))
		dataCipherSlice = append(dataCipherSlice, new(ckks.Ciphertext))
		plaintextOmega = append(plaintextOmega, []complex128{})
		plaintextTheta = append(plaintextTheta, []complex128{})
	}

	// ------- encode & encrypt ------- //
	// cut Slice and parallel two task weight and intercept to single complex128
	for i := 0; i < 1020; i++ {
		for num := 0; num < len(xlasso.plainOmega0[i]); num++ {
			omega := complex(xlasso.plainOmega0[i][num], xlasso.plainOmega1[i][num])
			plaintextOmega[i] = append(plaintextOmega[i], omega)
		}
		for num := 0; num < len(xlasso.plainTheta0[i]); num++ {
			theta := complex(xlasso.plainTheta0[i][num], xlasso.plainTheta1[i][num])
			plaintextTheta[i] = append(plaintextTheta[i], theta)
		}
	}
	for i := 0; i < 198; i++ {
		intercept := complex(xlasso.plainInterSlice0[i], xlasso.plainInterSlice1[i])
		plaintextInterSlice = append(plaintextInterSlice, intercept)
	}
	// encrypt
	time0 := time.Now()
	ciphertextInter := params.Encrypt(plaintextInterSlice)
	for i := 0; i < xlasso.modelParams.threadNum; i++ {
		go func(threadId int) {
			for batchNum := 0; batchNum < threadBatch; batchNum++ {
				encoder := params.encoder.ShallowCopy()
				encryptor := params.encryptor.ShallowCopy()
				encryptorSk := params.encryptorSk.ShallowCopy()
				omegaPlain := encoder.EncodeSlotsNew(plaintextOmega[threadId*threadBatch+batchNum], params.params.MaxLevel(), params.params.DefaultScale(), params.params.LogSlots())
				thetaPlain := encoder.EncodeSlotsNew(plaintextTheta[threadId*threadBatch+batchNum], params.params.MaxLevel(), params.params.DefaultScale(), params.params.LogSlots())
				dataPlain := encoder.EncodeSlotsNew(plainDataSlice[threadId*threadBatch+batchNum], params.params.MaxLevel(), params.params.DefaultScale(), params.params.LogSlots())
				omegaCipherSlice[threadId*threadBatch+batchNum] = encryptor.EncryptNew(omegaPlain)
				thetaCipherSlice[threadId*threadBatch+batchNum] = encryptor.EncryptNew(thetaPlain)
				dataCipherSlice[threadId*threadBatch+batchNum] = encryptorSk.EncryptNew(dataPlain)
			}

			wg.Done()
		}(i)
	}
	wg.Wait()
	wg.Add(xlasso.modelParams.threadNum)
	timeEncodeEncrypt = time.Since(time0)

	time1 := time.Now()
	for i := 0; i < xlasso.modelParams.threadNum; i++ {
		go func(threadId int) {
			for batchNum := 0; batchNum < threadBatch; batchNum++ {
				cipherThetaX, cipherOmegaX2 := params.XGBLassoThreadMul(omegaCipherSlice[threadId*threadBatch+batchNum], thetaCipherSlice[threadId*threadBatch+batchNum], dataCipherSlice[threadId*threadBatch+batchNum])
				lock.Lock()
				cipherThetaXSlice = append(cipherThetaXSlice, cipherThetaX)
				cipherOmegaX2Slice = append(cipherOmegaX2Slice, cipherOmegaX2)
				lock.Unlock()
			}
			wg.Done()
		}(i)
	}
	wg.Wait()
	timeDotMul = time.Since(time1)

	cipherRes := params.XGBLassoDotProduct(cipherThetaXSlice, cipherOmegaX2Slice)
	ciphertext := params.Add(cipherRes, ciphertextInter)

	// sigmoid function
	//if xlasso.modelParams.task0 > 3 {
	//	ciphertext = params.Sigmoid(ciphertext, xlasso.modelParams.encodeParams)
	//}

	time2 := time.Now()
	result := params.Decrypt(ciphertext)
	timeDecrypt = time.Since(time2)
	timeEvaluate = time.Since(timeAll)

	fmt.Printf("encode & encrypt time:%v || DotMul time:%v || relinearize time:%v || rescale time:%v || decrypt time: %v \n",
		timeEncodeEncrypt, timeDotMul+timeDotSum+timeDotAdd, timeRelin, timeRescale, timeDecrypt)
	fmt.Println("runtime: ", timeEvaluate)
	TraceMemStats()

	evaRes0 = ResFormat(result, xlasso.modelParams.task0, xlasso.modelParams.encodeParams, true)
	evaRes1 = ResFormat(result, xlasso.modelParams.task0, xlasso.modelParams.encodeParams, false)
	return
}
