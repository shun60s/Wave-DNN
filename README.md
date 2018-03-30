# 音声信号のDNN  

## 概要  

画像認識ではVGG16など事前学習したものを利用できるが、音声認識用途では少ない。 
そこで、音声認識エンジンJuliusのディクテーションキットに含まれるDNNを利用するための 
特徴量FBANK_D_A_Zを計算するpythonを作ってみた。 
  
## 使い方  
### 1.DNNのmodelの準備  
Juliusのディクテーションキットversion 4.4をダウンロードする。<http://julius.osdn.jp/index.php?q=dictation-kit.html>  
model/dnn以下を展開する。 
  

### 2.各プログラムの説明  
  
- get_fbank.py  16KHzサンプリングのwavファイルを読み込んで特徴量FBANK_D_A_Zを計算するクラス。
- cmvn_class.py  平均値・分散の正規化をするクラス。
- dnn_class.py  numpyでDNNを計算するクラス。
- chainer_dnn_class.py ディープラーニングのフレームワークのchainerでDNNを計算するクラス。学習は未対応。
- main0.py 16KHzサンプリングのwavファイルを読み込んで numpyでDNNを計算するまでのmainプログラムのサンプル。
- mainc.py 16KHzサンプリングのwavファイルを読み込んで chainerでDNNを計算するまでのmainプログラムのサンプル。
- bin/common/dnnclient.py dnn計算の入出力データをnpyファイルで書き出す変更をしたもの。



## 注意  
DNNの計算出力はHMMの隠れ状態の確率なので、そのままでは認識に使えない。   
計算出力は状態の事前確率で除算してLOG10をとった値になっている。  
  
> HMMは実質3状態のLR型で，4,874個の状態からなる状態共有モデルである． 
> 状態確率がDNNによって与えられる． 
  
Julius(C言語float)とpythonの数値計算の精度が同じではないので、計算結果は完全には一致しない。 
(およそ1.0E-5オーダー程度の差が出るようだ。)   
  
DNN-HMMの対数尤度の計算については[Wave-DNN-likelihood](https://github.com/shun60s/Wave-DNN-likelihood) を参照してください。   
  
## ライセンス  
以下のライセンス文を参照のこと。   
LICENSE-Julius Dictation Kit.txt  
LICENSE-Julius.txt  
LICENSE-PyHTK  





