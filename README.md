# cutfのsign関数の速度比較
変数の符号を返す関数`cutf::cuda::math::sign`の計算速度を調査する．

## 対象関数の特徴
- 分岐を用いずbit演算のみでの実装
- インラインアセンブラによりPTXで実装

## 実験
- 分岐を用いたsign関数実装と速度を比較
- half, float, doubleでテスト

### 実験環境
- CUDA 10.0
- nvprofで実行時間を測定

### 実験結果

- NVIDIA Titan V
```
Time(%) Time     Calls       Avg       Min       Max  Name
17.51%  25.564ms      8192  3.1200us  3.1030us  9.5670us  void kernel_if<double>(double const *, double*)
17.43%  25.440ms      8192  3.1050us  3.0710us  12.544us  void kernel_if<__half>(__half const *, __half*)
17.37%  25.359ms      8192  3.0950us  3.0710us  10.271us  void kernel_if<float>(float const *, float*)
16.12%  23.523ms      8192  2.8710us  2.7830us  12.576us  void kernel_cutf<__half>(__half const *, __half*)
15.79%  23.042ms      8192  2.8120us  2.7830us  10.495us  void kernel_cutf<double>(double const *, double*)
15.77%  23.016ms      8192  2.8090us  2.7830us  9.7280us  void kernel_cutf<float>(float const *, float*)
```

- NVIDIA Geforce GTX 1080Ti
```
Time(%)      Time     Calls       Avg       Min       Max  Name
44.41%  127.21ms      8192  15.528us  13.184us  28.481us  void kernel_if<__half>(__half const *, __half*)
22.12%  63.363ms      8192  7.7340us  3.8720us  12.449us  void kernel_if<float>(float const *, float*)
9.98%  28.586ms      8192  3.4890us  2.8800us  4.6410us  void kernel_if<double>(double const *, double*)
7.99%  22.895ms      8192  2.7940us  2.6880us  3.2640us  void kernel_cutf<__half>(__half const *, __half*)
7.84%  22.463ms      8192  2.7420us  2.7200us  3.1360us  void kernel_cutf<float>(float const *, float*)
7.64%  21.896ms      8192  2.6720us  2.4320us  3.1040us  void kernel_cutf<double>(double const *, double*)
```

|型    |高速化(TitanV)|高速化(1080Ti)|
|:----:|:------------:|:------------:|
|half  | x1.08        | x5.56        |
|float | x1.10        | x2.82        |
|double| x1.10        | x1.30        |

