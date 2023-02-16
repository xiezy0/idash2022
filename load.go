package main

import (
	"bufio"
	"encoding/csv"
	"fmt"
	"os"
	"strconv"
	"strings"
)

// LoadWeight 权重加载
func LoadWeight(csvFile string, encodeParams float64) [][]float64 {
	file, err := os.Open(csvFile)
	if err != nil {
		fmt.Println(err)
	}

	defer func(file *os.File) {
		err := file.Close()
		if err != nil {
			panic(err)
		}
	}(file)

	reader := csv.NewReader(file)
	reader.FieldsPerRecord = -1
	csvData, err := reader.ReadAll()
	w1slice := make([]float64, 0)
	for _, line := range csvData {
		dataRead, _ := strconv.ParseFloat(line[0], 64)
		w1 := fmt.Sprintf("%.5f", dataRead)
		w11, _ := strconv.ParseFloat(w1, 64)
		w1slice = append(w1slice, w11*encodeParams) // 读取以后进行编码*encodeParams
	}

	var plaintextWeights [][]float64
	num := 0

	// 创建前20380条基因的权重切片 共1019个块, 将3960条数据打包进一个明文块：[(198),(198),......,(198),(198)]
	//time0 := time.Now()
	for plaintextNum := 0; plaintextNum < 1019; plaintextNum++ {
		plaintextWeight := make([]float64, 0)
		for slotNum := 0; slotNum < 20; slotNum++ {
			weight := make([]float64, 0)
			// 重复的weight切片创建
			for i := 0; i < 198; i++ {
				weight = append(weight, w1slice[num])
			}
			plaintextWeight = append(plaintextWeight, weight...) // 将重复的切片拼接到一起
			num++
		}
		plaintextWeights = append(plaintextWeights, plaintextWeight)
	}

	// 创建后10条基因的权重切片 共1个, 将1980条数据打包进一个明文块：[(198),(198),......,(198),(198)]
	plaintextWeight := make([]float64, 0)
	for slotNum := 0; slotNum < 10; slotNum++ {
		weight := make([]float64, 0)
		// 重复的weight切片创建
		for i := 0; i < 198; i++ {
			weight = append(weight, w1slice[num])
		}

		plaintextWeight = append(plaintextWeight, weight...) // 将重复的切片拼接到一起
		num++
	}
	plaintextWeights = append(plaintextWeights, plaintextWeight)
	//fmt.Println("ttt", time.Since(time0))

	return plaintextWeights
}

// LoadData 数据加载
func LoadData() [][]float64 {
	file, _ := os.Open("data/genotypes.txt")
	scanner := bufio.NewScanner(file)
	geneNum := 0
	plaintextData := make([]float64, 0)
	var plaintextDataSlice [][]float64
	// 读取20390行的前198列条数据
	for {
		if !scanner.Scan() {
			plaintextDataSlice = append(plaintextDataSlice, plaintextData)
			break
		}
		dataI := String2Float64(strings.Fields(scanner.Text())[4:202])
		geneNum++
		plaintextData = append(plaintextData, dataI...)
		if geneNum == 20 {
			geneNum = 0
			plaintextDataSlice = append(plaintextDataSlice, plaintextData)
			plaintextData = make([]float64, 0)
		}
	}
	return plaintextDataSlice
}

// LoadIntercept 截距加载
func LoadIntercept(encodeParams float64) [][]float64 {
	file, err := os.Open("intercept/intercept.csv")
	if err != nil {
		fmt.Println(err)
	}

	var interceptSlices [][]float64
	defer func(file *os.File) {
		err := file.Close()
		if err != nil {

		}
	}(file)

	reader := csv.NewReader(file)
	reader.FieldsPerRecord = -1
	csvData, err := reader.ReadAll()

	for _, line := range csvData {
		interceptSlice := make([]float64, 0)
		dataRead, _ := strconv.ParseFloat(line[0], 64)
		w1 := fmt.Sprintf("%.5f", dataRead)
		w11, _ := strconv.ParseFloat(w1, 64)
		for i := 0; i < 198; i++ {
			interceptSlice = append(interceptSlice, w11*encodeParams) // 编码 * encodeParams
		}
		interceptSlices = append(interceptSlices, interceptSlice)
	}

	return interceptSlices
}

// LoadLabel Label加载
func LoadLabel(csvFile string) []float64 {
	file, err := os.Open(csvFile)
	if err != nil {
		fmt.Println(err)
	}
	defer func(file *os.File) {
		err := file.Close()
		if err != nil {
			panic(err)
		}
	}(file)

	reader := csv.NewReader(file)
	reader.FieldsPerRecord = -1
	csvData, err := reader.ReadAll()
	w1slice := make([]float64, 0)
	for i, line := range csvData {
		dataRead, _ := strconv.ParseFloat(line[0], 64)
		w1 := fmt.Sprintf("%.10f", dataRead)
		w11, _ := strconv.ParseFloat(w1, 64)
		w1slice = append(w1slice, w11)
		if i == 197 {
			break
		}
	}
	return w1slice
}

// LoadWeightPlain 权重加载，用于明文计算
func LoadWeightPlain(csvFile string) []float64 {
	file, err := os.Open(csvFile)
	if err != nil {
		fmt.Println(err)
	}
	defer func(file *os.File) {
		err := file.Close()
		if err != nil {
			panic(err)
		}
	}(file)

	reader := csv.NewReader(file)
	reader.FieldsPerRecord = -1
	csvData, err := reader.ReadAll()
	w1slice := make([]float64, 0)
	for _, line := range csvData {
		dataRead, _ := strconv.ParseFloat(line[0], 64)
		w1 := fmt.Sprintf("%.5f", dataRead)
		w11, _ := strconv.ParseFloat(w1, 64)
		w1slice = append(w1slice, w11)
	}
	return w1slice
}

// LoadDataPlain 数据加载，用于明文计算
func LoadDataPlain(num int) []float64 {
	file, _ := os.Open("data/genotypes.txt")
	scanner := bufio.NewScanner(file)
	plaintextData := make([]float64, 0)
	// 读取20390行的前198列条数据
	for {
		if !scanner.Scan() {
			break
		}
		d11, _ := strconv.ParseFloat(strings.Fields(scanner.Text())[num+3], 64)
		plaintextData = append(plaintextData, d11)
	}
	return plaintextData
}

// LoadDataRewrite 数据重写，缩小文件大小
func LoadDataRewrite() [][]float64 {
	file, _ := os.Open("data/genotypes.txt")
	fileWrite, _ := os.OpenFile("data/test.txt", os.O_APPEND, 0666)
	scanner := bufio.NewScanner(file)
	var plaintextDataSlice [][]float64
	// 读取20390行的前198列条数据
	for {
		if !scanner.Scan() {
			break
		}
		_, err := fileWrite.WriteString(convertStringsToBytes(strings.Fields(scanner.Text())[4:202]) + "\n")
		if err != nil {
			return nil
		}
	}
	return plaintextDataSlice
}

// LoadDataRenew 读取重写的文件
func LoadDataRenew() [][]float64 {
	file, _ := os.Open("data/test.txt")
	scanner := bufio.NewScanner(file)
	geneNum := 0
	plaintextData := make([]float64, 0)
	var plaintextDataSlice [][]float64
	// 读取20390行的前198列条数据
	for {
		if !scanner.Scan() {
			plaintextDataSlice = append(plaintextDataSlice, plaintextData)
			break
		}
		dataI := String2Float64(strings.Fields(scanner.Text())[0:198])
		geneNum++
		plaintextData = append(plaintextData, dataI...)
		if geneNum == 20 {
			geneNum = 0
			plaintextDataSlice = append(plaintextDataSlice, plaintextData)
			plaintextData = make([]float64, 0)
		}
	}
	return plaintextDataSlice
}

func convertStringsToBytes(stringContent []string) string {
	ss := fmt.Sprintf(strings.Join(stringContent, " "))
	return ss
}
