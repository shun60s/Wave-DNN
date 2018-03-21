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
  
- get_fbank.py  wavファイルを読み込んで特徴量FBANK_D_A_Zを計算するクラス。
- cmvn_class.py  平均値・分散の正規化をするクラス。
- dnn_class.py  DNNを計算するクラス。
- main0.py 1つのwavファイルを読み込んで dnnを計算するまでのmainプログラムのサンプル。
- bin/common/dnnclient.py dnn計算の入出力データをnpyファイルで書き出す変更をしたもの。



## 注意  
DNNの計算出力はHMMの隠れ状態の確率なので　そのままの形では認識に応用はできない。 
  
> HMMは実質3状態のLR型で，4,874個の状態からなる状態共有モデルである． 
> 状態確率がDNNによって与えられる． 
  
Julius(C言語float)とpythonの数値計算の精度が同じではないので、計算結果は完全には一致しない。 
(およそ1.0E-5オーダー程度の差が出るようだ。) 

## ライセンス  
以下のライセンス文を参照のこと。 
LICENSE-Julius Dictation Kit.txt  
LICENSE-Julius.txt  
LICENSE-PyHTK  





