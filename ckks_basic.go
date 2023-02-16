package main

import (
	"math"
	"sync"

	"github.com/tuneinsight/lattigo/v3/ckks"
	"github.com/tuneinsight/lattigo/v3/rlwe"
)

// Ckksparams 封装CKKS功能
type Ckksparams struct {
	params      ckks.Parameters
	encoder     ckks.Encoder
	encryptorSk ckks.Encryptor
	encryptor   ckks.Encryptor
	decryptor   ckks.Decryptor
	evaluator   ckks.Evaluator
}

type Ckks interface {
	Encrypt(input interface{}) (Ciphertext *ckks.Ciphertext)                                                                        // 加密
	Add(Ciphertext0 *ckks.Ciphertext, Ciphertext1 *ckks.Ciphertext) (CiphertextAdd *ckks.Ciphertext)                                // 加法同态
	AddThread(encInputs []*ckks.Ciphertext) *ckks.Ciphertext                                                                        // 多线程加法同态
	Mul(Ciphertext0 *ckks.Ciphertext, Ciphertext1 *ckks.Ciphertext) (CiphertextMult *ckks.Ciphertext)                               // 乘法同态 + 线性化 + 缩放
	MulNoRelin(Ciphertext0 *ckks.Ciphertext, Ciphertext1 *ckks.Ciphertext) (CiphertextMult *ckks.Ciphertext)                        // 乘法同态
	InnerSum(Ciphertext0 *ckks.Ciphertext, Ciphertext1 *ckks.Ciphertext, slot float64) (CiphertextMult *ckks.Ciphertext)            // 内积
	InnerSumBlock(Ciphertext0 *ckks.Ciphertext, Ciphertext1 *ckks.Ciphertext, Block int, slot int) (CiphertextRot *ckks.Ciphertext) // 内积 块=198
	Decrypt(Ciphertext *ckks.Ciphertext) (output []complex128)                                                                      // 解密
}

// GenKey CKKS密钥生成
func GenKey(paramDef ckks.ParametersLiteral) Ckksparams {
	params, err := ckks.NewParametersFromLiteral(paramDef)
	if err != nil {
		panic(err)
	}
	kGen := ckks.NewKeyGenerator(params)
	// 公私钥生成
	sk, pk := kGen.GenKeyPair()
	// 重线性化密钥
	rlk := kGen.GenRelinearizationKey(sk, 3)
	// 加密器
	encryptorSk := ckks.NewEncryptor(params, sk)
	encryptor := ckks.NewEncryptor(params, pk)
	// 解密器
	decryptor := ckks.NewDecryptor(params, sk)
	// 编码器
	encoder := ckks.NewEncoder(params)
	// 旋转密钥
	rots := make([]int, 0)
	for i := 0; i < 5; i++ {
		rots = append(rots, int(math.Pow(2, float64(i)))*198)
	}
	//rots := make([]int, 0)
	//rots = append(rots, 198*2)
	rotKey := kGen.GenRotationKeysForRotations(rots, true, sk)
	//同态计算器
	evaluator := ckks.NewEvaluator(params, rlwe.EvaluationKey{Rlk: rlk, Rtks: rotKey})

	return Ckksparams{params, encoder, encryptorSk, encryptor, decryptor, evaluator}
}

func (params *Ckksparams) Encrypt(input interface{}) (Ciphertext *ckks.Ciphertext) {
	plaintext := params.encoder.EncodeSlotsNew(input, params.params.MaxLevel(), params.params.DefaultScale(), params.params.LogSlots())
	Ciphertext = params.encryptor.EncryptNew(plaintext)
	return
}

func (params *Ckksparams) Add(Ciphertext0 *ckks.Ciphertext, Ciphertext1 *ckks.Ciphertext) (CiphertextAdd *ckks.Ciphertext) {
	CiphertextAdd = params.evaluator.AddNew(Ciphertext0, Ciphertext1)
	return
}

func (params *Ckksparams) AddThread(encInputs []*ckks.Ciphertext) *ckks.Ciphertext {

	var wg sync.WaitGroup

	encLvls := make([][]*ckks.Ciphertext, 0)
	encLvls = append(encLvls, encInputs)
	for nLvl := (len(encInputs) + 1) / 2; nLvl >= 1; nLvl = nLvl / 2 {
		encLvl := make([]*ckks.Ciphertext, nLvl)
		for i := range encLvl {
			encLvl[i] = ckks.NewCiphertext(params.params, 1, params.params.MaxLevel(), params.params.DefaultScale())
		}
		encLvls = append(encLvls, encLvl)
		if nLvl%2 != 0 && nLvl != 1 {
			nLvl++
		}
	}

	for i := 0; i < len(encLvls)-1; i++ {
		wg.Add(len(encLvls[i+1]))
		for j := 0; j < len(encLvls[i+1]); j++ {
			go func(ThreadId int, RoundId int) {
				if len(encLvls[i])%2 != 0 && ThreadId == len(encLvls[i+1])-1 {
					encLvls[RoundId+1][ThreadId] = encLvls[RoundId][ThreadId*2]
					wg.Done()
				} else {
					AddCopy := params.evaluator.ShallowCopy()
					encLvls[RoundId+1][ThreadId] = AddCopy.AddNew(encLvls[RoundId][ThreadId*2], encLvls[RoundId][ThreadId*2+1])
					wg.Done()
				}

			}(j, i)
		}
		wg.Wait()
	}

	return encLvls[len(encLvls)-1][0]
}

func (params *Ckksparams) MulNoRelin(Ciphertext0 *ckks.Ciphertext, Ciphertext1 *ckks.Ciphertext) (CiphertextMul *ckks.Ciphertext) {
	CiphertextMul = params.evaluator.MulNew(Ciphertext0, Ciphertext1)
	return
}

func (params *Ckksparams) Mul(Ciphertext0 *ckks.Ciphertext, Ciphertext1 *ckks.Ciphertext) (CiphertextMul *ckks.Ciphertext) {
	CiphertextMul = params.evaluator.MulRelinNew(Ciphertext0, Ciphertext1)
	err := params.evaluator.Rescale(CiphertextMul, params.params.DefaultScale(), CiphertextMul)
	if err != nil {
		panic("Mul error")
		return nil
	}
	return
}

func (params *Ckksparams) InnerSum(Ciphertext0 *ckks.Ciphertext, Ciphertext1 *ckks.Ciphertext, slot float64) (CiphertextOut *ckks.Ciphertext) {
	CiphertextOut = params.Mul(Ciphertext0, Ciphertext1)
	//fmt.Println(params.Decrypt(CiphertextOut))
	ployDegree := math.Log2(slot)
	// fmt.Println(ployDegree)
	//fmt.Println(params.Decrypt(CiphertextOut))
	for i := 0; i < int(ployDegree); i++ {
		ciphertextNew := params.evaluator.RotateNew(CiphertextOut, int(math.Pow(2, float64(i))))
		params.evaluator.Add(ciphertextNew, CiphertextOut, CiphertextOut)
	}
	return
}

func (params *Ckksparams) InnerSumBlock(Ciphertext0 *ckks.Ciphertext, Ciphertext1 *ckks.Ciphertext, Block int, slot int) (CiphertextRot *ckks.Ciphertext) {
	CiphertextRot = params.Mul(Ciphertext0, Ciphertext1)
	params.evaluator.InnerSumLog(CiphertextRot, Block, slot, CiphertextRot)
	//CiphertextOut := CiphertextRot.CopyNew()
	//for i := 1; i < int(slot); i++ {
	//	params.evaluator.Rotate(CiphertextRot, 198, CiphertextRot)
	//	//if i == 0 {
	//	//	fmt.Println(params.Decrypt(CiphertextRot))
	//	//}
	//	params.evaluator.Add(CiphertextOut, CiphertextRot, CiphertextOut)
	//}
	return CiphertextRot
}

func (params *Ckksparams) Decrypt(Ciphertext *ckks.Ciphertext) (output []complex128) {
	Plaintext := params.decryptor.DecryptNew(Ciphertext)
	output = params.encoder.DecodeSlots(Plaintext, params.params.LogSlots())
	return
}
