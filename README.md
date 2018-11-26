# cutfのsign関数の速度比較
変数の符号を返す関数`cutf::cuda::math::sign`の計算速度を調査する．

## 対象関数の特徴
- 分岐を用いずbit演算のみでの実装
- インラインアセンブラによりPTXで実装

## 実験
- 分岐を用いたsign関数実装と速度を比較
- half, float, doubleでテスト

### 実験環境
- NVIDIA Titan V
- CUDA 10.0

### 実験結果
- nvprofで実行時間を測定

```
Time     Calls       Avg       Min       Max  Name
17.51%  25.564ms      8192  3.1200us  3.1030us  9.5670us  void kernel_if<double>(double const *, double*)
17.43%  25.440ms      8192  3.1050us  3.0710us  12.544us  void kernel_if<__half>(__half const *, __half*)
17.37%  25.359ms      8192  3.0950us  3.0710us  10.271us  void kernel_if<float>(float const *, float*)
16.12%  23.523ms      8192  2.8710us  2.7830us  12.576us  void kernel_cutf<__half>(__half const *, __half*)
15.79%  23.042ms      8192  2.8120us  2.7830us  10.495us  void kernel_cutf<double>(double const *, double*)
15.77%  23.016ms      8192  2.8090us  2.7830us  9.7280us  void kernel_cutf<float>(float const *, float*)
```

|型    |高速化|
|:----:|:----:|
|half  | x1.08|
|float | x1.10|
|double| x1.10|

少しだけ速くなりましたとさ
