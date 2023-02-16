# Idash_Otriver

版本号：`v2.0.0 `
dev分支

简介：基于`Lattigo v3.0.5` 实现密文下的线性与非线性模型评估

环境配置：

- 系统环境：`windows 10`及以上或`ubuntu 18`及以上系统。
- GO SDK：golang 1.18
- 第三方库链接：https://github.com/tuneinsight/lattigo

## 1 效率测试

计算用时包括（密钥生成，加解密，编解码，同态计算）

测试环境（Starfire: 4 Core Intel(R) Xeon(R) Platinum 8269CY CPU @ 2.50GHz，32g内存）

|             |   Lasso   | XGB & Lasso | Lasso（Two Task） | XGB & Lasso（Two Task） |
| :---------: | :-------: | :---------: | :---------------: | :---------------------: |
|  数据加载   | 460.51 ms |  500.31 ms  |     602.06 ms     |       811. 92 ms        |
| 编码 & 加密 |  2.74 s   |   4.54 s    |      2.72 s       |         4.61 s          |
|    点积     |  1.00 s   |   1.97 s    |      1.05 s       |         1.95 s          |
|  重线性化   |  4.96 ms  |  13.35 ms   |      4.69 ms      |        15.12 ms         |
|   重缩放    |  1.03 ms  |   2.77 ms   |      1.36 ms      |         2.90 ms         |
|    解密     |  8.93 ms  |   8.26 ms   |      8.59 ms      |         8.67 ms         |
|  计算用时   |  4.01 s   |   6.77 s    |      4.15 s       |         7.07 s          |

## 2 准确率测试

线性逻辑回归模型的准确率：

|        | RMSE(非编码) | AUC(非编码) |
| :----: | :----------: | :---------: |
| Task 1 |   0.01966    |             |
| Task 2 |   0.03814    |             |
| Task 3 |   0.05155    |             |
| Task 4 |              |   0.98990   |
| Task 5 |              |   0.98990   |

## 3 用法

### 3.1 运行方法

源码下载与依赖拉取

```shell
git clone http://gitlab.os.adc.com/OppoResearch/dataintel/privacy/idash_otriver.git
```

注：go mod 命令在starfire平台上可能无法联网使用，需要设置GOPROXY变量

运行`./utils_test.go`下面的五个测试TASK，对五个任务进行测试。

```shell
sudo go build -buildvcs=false
sudo go test -c utils_test.go utils.go load.go ckks_basic.go model.go linear.go nonlinear.go -o evaluate.test
./evaluate.test
```

### 3.2 用例

- 线性模型 + 实部编码

```go
// model = 1(linear model), task = 1(continuous task), thread = 6(6 goroutine), encodeParams = 1(weight * encodeParams)
params := ModelParams{1, 1, 6, 1}
evaRes := params.Model()
label := LoadLabel("label/label_1.csv")
RMSE := CompareRMSE(evaRes, label)
fmt.Println("TASK 1 RMSE: ", RMSE)

// model = 1(linear model), task = 4(discrete task), thread = 4(4 goroutine), encodeParams = 1(weight * encodeParams)
params := ModelParams{1, 4, 4, 1}
evaRes := params.Model()
label := LoadLabel("label/label_4.csv")
AUC := CompareAUC(evaRes, label)
fmt.Println("TASK 4 AUC: ", AUC)
```

- 非线性模型 + 实部编码

```go
// model = 2(nonlinear model), task = 1(continuous task), thread = 6(6 goroutine), encodeParams = 1(weight * encodeParams)
params := ModelParams{2, 1, 6, 1}
evaRes := params.Model()
label := LoadLabel("label/label_1.csv")
RMSE := CompareRMSE(evaRes, label)
fmt.Println("RMSE: ", RMSE)

// model = 2(nonlinear model), task = 4(discrete task), thread = 6(6 goroutine), encodeParams = 1(weight * encodeParams)
params := ModelParams{2, 4, 6, 1}
evaRes := params.Model()
label := LoadLabel("label/label_4.csv")
AUC := CompareAUC(evaRes, label)
fmt.Println("TASK 4 AUC: ", AUC)
```

- 复数编码 （将任务1的权重编码到实部，将任务5的权重编码到虚部）

```go
// model = 1(linear model), task0 = 1(continuous task), thread = 6(6 goroutine), task1 = 5(discrete task) encodeParams = 1(weight * encodeParams)
params := ModelParallelParams{1, 1, 5, 6, 1}
evaRes0, evaRes1 := params.ModelParallel()
label1 := LoadLabel("label/label_1.csv")
RMSE1 := CompareRMSE(evaRes0, label1)
fmt.Println("TASK 1 RMSE: ", RMSE1)

label5 := LoadLabel("label/label_5.csv")
AUC5 := CompareAUC(evaRes1, label5)
fmt.Println("TASK 5 AUC: ", AUC5)
```

## 5 底层库更改

### 5.1 模数链新增

SEAL参数新增

文件：`./vendor/github.com/tuneinsight/lattigo/v3/ckks/params.go`

```go
// choose params `SEAL BFV-128bit-default` which same as SEAL 
PN13QP218SEAL = ParametersLiteral{
		LogN:     13,
		LogSlots: 12,
		Q: []uint64{0x7fffffc8001, // 43 + 3 x 44
			0xfffffffc001,
			0xffffff6c001,
			0xfffffebc001},
		P:            []uint64{0x7fffffd8001}, // 43
		DefaultScale: 1 << 44,
		Sigma:        rlwe.DefaultSigma,
		RingType:     ring.Standard,
}
```

### 5.2 多维同态乘法

1维密文与2维密文的乘法

文件：`./vendor/github.com/tuneinsight/lattigo/v3/ckks/evaluator.go`

```go
// add Interface
Mul2DegreeNew(op0, op1 Operand) (ctOut *Ciphertext)
```

### 5.3 多维密文重线性化

3维密文重线性化

文件：`./vendor/github.com/tuneinsight/lattigo/v3/ckks/evaluator.go`

```go
// add Interface
RelinearizeDegree(ct0 *Ciphertext, ctOut *Ciphertext)
```

