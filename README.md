# rlDiscreteLQT

## クラスの仕様

* インスタンス
  * $x_{k+1} = A_d x_k + B_d u_k$
  * x0を指定しない場合はreset実行時に$x0 \in [0,1]$となるように変更
